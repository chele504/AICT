from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUESTION_COUNT = 20
QUESTION_PREFIX_RE = re.compile(r"^\s*(\d{1,2})\s*[.、．]")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


DEFAULT_SCORING = {
    "input_value_order": "reverse_order",
    "dimensions": {
        "tech_empowerment": {
            "items": [1, 2, 3, 4, 5],
            "weight": 0.25,
            "name_cn": "技术赋能效能",
        },
        "visitor_experience": {
            "items": [6, 7, 8, 9, 10],
            "weight": 0.30,
            "name_cn": "游客感知体验",
        },
        "cultural_value": {
            "items": [11, 12, 13, 14, 15],
            "weight": 0.25,
            "name_cn": "文化价值传播",
        },
        "economic_social_gain": {
            "items": [16, 17, 18, 19, 20],
            "weight": 0.20,
            "name_cn": "经济社会增值",
        },
    },
}

DEFAULT_QUALITY = {
    "min_duration_seconds": 60,
    "low_personal_std_threshold": 0.5,
    "overall_inconsistency_threshold": 2.0,
    "valid_score_threshold": 60,
    "review_score_threshold": 75,
    "penalties": {
        "short_duration": 15,
        "straight_line": 15,
        "low_personal_variance": 10,
        "duplicate_answer_pattern": 10,
        "duplicate_ip": 5,
        "overall_inconsistency": 15,
    },
    "placeholder_texts": [
        "",
        "(空)",
        "空",
        "无",
        "没有",
        "暂无",
        "没有不舒适",
        "无明显不适",
        "无需改进",
    ],
}

DEFAULT_AI = {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com/chat/completions",
    "model": "deepseek-v4-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "timeout_seconds": 60,
    "max_retries": 3,
    "temperature": 0,
    "thinking_enabled": True,
    "reasoning_effort": "high",
    "max_tokens": 2000,
    "request_interval_seconds": 0.2,
}

DEFAULT_SPLIT = {"test_ratio": 0.20, "random_seed": 42}
DEFAULT_PRIVACY = {"retain_raw_ip": False}
DEFAULT_MODEL_DATASET = {"include_question_items": False, "include_derived_dimensions": False}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("unknown", b"", 0, 1, "; ".join(errors))


def canonical_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def repair_duration_text(value: Any) -> str:
    text = canonical_text(value)
    if not text:
        return ""
    for source_encoding, target_encoding in (("cp1251", "gb18030"), ("gb18030", "utf-8")):
        try:
            candidate = text.encode(source_encoding).decode(target_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if any(token in candidate for token in ("秒", "分", "分钟", "小时")):
            return candidate
    return text


def parse_duration_seconds(value: Any) -> float:
    text = repair_duration_text(value)
    if not text:
        return np.nan
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:小时|时)", text)
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:分钟|分)", text)
    second_match = re.search(r"(\d+(?:\.\d+)?)\s*秒", text)
    if hour_match or minute_match or second_match:
        hours = float(hour_match.group(1)) if hour_match else 0.0
        minutes = float(minute_match.group(1)) if minute_match else 0.0
        seconds = float(second_match.group(1)) if second_match else 0.0
        return hours * 3600 + minutes * 60 + seconds
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numbers[0]) if len(numbers) == 1 else np.nan


def normalize_likert(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        text = canonical_text(value)
        labels = {
            "非常不同意": 1,
            "不同意": 2,
            "一般": 3,
            "同意": 4,
            "非常同意": 5,
        }
        exact = labels.get(text)
        if exact is not None:
            return float(exact)
        match = re.search(r"(?<!\d)([1-5])(?:\.0)?(?:\s*分)?(?!\d)", text)
        if not match:
            return np.nan
        number = float(match.group(1))
    return number if number in {1.0, 2.0, 3.0, 4.0, 5.0} else np.nan


def question_candidate_priority(column: str, number: int) -> int:
    lowered = column.strip().lower()
    if lowered == f"q{number}":
        return 4
    if lowered == f"raw_q{number}":
        return 3
    if QUESTION_PREFIX_RE.match(column):
        return 2
    return 0


def detect_columns(df: pd.DataFrame) -> tuple[dict[int, str], str | None, dict[str, str | None]]:
    candidates: dict[int, list[tuple[int, str]]] = {number: [] for number in range(1, 22)}
    for column in df.columns:
        lowered = str(column).strip().lower()
        exact_match = re.fullmatch(r"(?:raw_)?q(\d{1,2})", lowered)
        prefix_match = QUESTION_PREFIX_RE.match(str(column))
        if exact_match:
            number = int(exact_match.group(1))
        elif prefix_match:
            number = int(prefix_match.group(1))
        else:
            continue
        if 1 <= number <= 21:
            candidates[number].append((question_candidate_priority(str(column), number), str(column)))

    question_columns: dict[int, str] = {}
    for number in range(1, QUESTION_COUNT + 1):
        if candidates[number]:
            question_columns[number] = max(candidates[number], key=lambda item: item[0])[1]

    text_column = None
    if "review_text" in df.columns:
        text_column = "review_text"
    elif candidates[21]:
        text_column = max(candidates[21], key=lambda item: item[0])[1]

    aliases = {
        "record_id": ["序号", "编号", "respondent_id", "record_id"],
        "submitted_at": ["提交答卷时间", "提交时间", "submitted_at", "timestamp"],
        "duration": ["所用时间", "答题时长", "duration", "duration_seconds"],
        "source": ["来源", "source", "source_type"],
        "source_detail": ["来源详情", "source_detail"],
        "ip": ["来自IP", "IP", "ip", "ip_address"],
    }
    metadata: dict[str, str | None] = {}
    for key, names in aliases.items():
        metadata[key] = next((name for name in names if name in df.columns), None)
    return question_columns, text_column, metadata


def redact_text(text: str) -> str:
    text = PHONE_RE.sub("[手机号已脱敏]", text)
    text = ID_CARD_RE.sub("[身份证号已脱敏]", text)
    return EMAIL_RE.sub("[邮箱已脱敏]", text)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_feedback(value: Any, placeholders: set[str]) -> tuple[str, bool]:
    text = redact_text(canonical_text(value))
    compact = re.sub(r"[\s，。！？,.!?;；:：]+", "", text).lower()
    normalized_placeholders = {
        re.sub(r"[\s，。！？,.!?;；:：]+", "", item).lower() for item in placeholders
    }
    meaningful = bool(compact) and compact not in normalized_placeholders and len(compact) >= 4
    return (text if text else "无明显不适"), meaningful


def local_text_analysis(text: str, meaningful: bool) -> dict[str, Any]:
    if not meaningful:
        return {
            "valid_feedback": False,
            "has_discomfort": False,
            "sentiment": "no_feedback",
            "problem_types": [],
            "scene": "",
            "symptoms": [],
            "suggestion": "",
            "summary": "无具体负面反馈",
            "confidence": 1.0,
            "analysis_source": "rule",
        }

    category_keywords = {
        "性能稳定性": ["卡顿", "掉线", "加载", "响应慢", "崩溃", "延迟"],
        "内容质量": ["错误", "不准确", "内容少", "内容多", "重复", "讲解"],
        "易用性": ["不会用", "难找", "操作", "入口", "复杂"],
        "沉浸与真实性": ["真实感", "沉浸", "画面", "切换"],
        "隐私安全": ["隐私", "权限", "个人信息"],
        "无障碍与包容性": ["老人", "儿童", "残障", "语言", "字幕"],
        "身体不适": ["眩晕", "眼疲劳", "头痛", "恶心", "疲劳"],
        "心理负担": ["紧张", "焦虑", "信息过载", "被打扰", "烦"],
    }
    problem_types = [
        category for category, keywords in category_keywords.items() if any(word in text for word in keywords)
    ]
    symptom_words = [
        word
        for word in ("眩晕", "眼疲劳", "头痛", "恶心", "紧张", "焦虑", "信息过载", "被打扰")
        if word in text
    ]
    positive_words = ("很好", "满意", "不错", "很棒", "完美", "没有问题", "良好")
    negative_words = ("不适", "卡顿", "错误", "疲劳", "头痛", "担忧", "不好", "较差", "希望")
    positive = sum(word in text for word in positive_words)
    negative = sum(word in text for word in negative_words)
    sentiment = "negative" if negative > positive else "positive" if positive > negative else "neutral"
    suggestion = text if any(word in text for word in ("希望", "建议", "增加", "改进", "优化")) else ""
    return {
        "valid_feedback": True,
        "has_discomfort": bool(problem_types or symptom_words or negative),
        "sentiment": sentiment,
        "problem_types": problem_types,
        "scene": "",
        "symptoms": symptom_words,
        "suggestion": suggestion,
        "summary": text[:120],
        "confidence": 0.55,
        "analysis_source": "rule",
    }


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return canonical_text(value).lower() in {"true", "1", "yes", "是", "有"}


def coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [canonical_text(item) for item in value if canonical_text(item)]
    text = canonical_text(value)
    if not text:
        return []
    return [item for item in re.split(r"[|,，;；、]+", text) if item]


def normalize_ai_result(payload: dict[str, Any], source: str) -> dict[str, Any]:
    sentiment = canonical_text(payload.get("sentiment", "neutral")).lower()
    if sentiment not in {"negative", "neutral", "positive", "no_feedback"}:
        sentiment = "neutral"
    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "valid_feedback": coerce_bool(payload.get("valid_feedback", False)),
        "has_discomfort": coerce_bool(payload.get("has_discomfort", False)),
        "sentiment": sentiment,
        "problem_types": coerce_string_list(payload.get("problem_types", [])),
        "scene": canonical_text(payload.get("scene", "")),
        "symptoms": coerce_string_list(payload.get("symptoms", [])),
        "suggestion": canonical_text(payload.get("suggestion", "")),
        "summary": canonical_text(payload.get("summary", ""))[:120],
        "confidence": min(max(confidence, 0.0), 1.0),
        "analysis_source": source,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("AI 响应中没有 JSON 对象")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("AI 响应不是 JSON 对象")
    return payload


class DeepSeekAnalyzer:
    def __init__(self, config: dict[str, Any], cache_path: Path, enabled: bool) -> None:
        self.config = config
        self.cache_path = cache_path
        self.enabled = enabled
        self.api_key = os.getenv(config.get("api_key_env", "DEEPSEEK_API_KEY"), "").strip()
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        row = json.loads(line)
                        self.cache[row["text_hash"]] = row["result"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def analyze(self, text: str, meaningful: bool) -> dict[str, Any]:
        fallback = local_text_analysis(text, meaningful)
        if not meaningful or not self.enabled or not self.api_key:
            return fallback
        cache_identity = json.dumps(
            {
                "text": text,
                "model": self.config.get("model", "deepseek-v4-flash"),
                "thinking_enabled": bool(self.config.get("thinking_enabled", False)),
                "reasoning_effort": self.config.get("reasoning_effort", "high"),
                "prompt_version": "questionnaire-label-v2",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        key = text_hash(cache_identity)
        if key in self.cache:
            return self.cache[key]

        system_prompt = (
            "你是文化旅游问卷数据标注助手。只分析游客开放题，不推断身份，不修改量表分数。"
            "必须输出一个合法的 json 对象，不要输出 markdown。字段为 valid_feedback(bool)、has_discomfort(bool)、"
            "sentiment(negative/neutral/positive)、problem_types(string数组)、scene(string)、"
            "symptoms(string数组)、suggestion(string)、summary(string，不超过80字)、confidence(0到1)。"
            "示例 JSON：{\"valid_feedback\":true,\"has_discomfort\":true,"
            "\"sentiment\":\"negative\",\"problem_types\":[\"性能稳定性\"],"
            "\"scene\":\"智能导览\",\"symptoms\":[],\"suggestion\":\"优化响应速度\","
            "\"summary\":\"导览偶有卡顿\",\"confidence\":0.9}"
        )
        payload = {
            "model": self.config.get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分析这条已脱敏反馈：\n{text}"},
            ],
            "temperature": self.config.get("temperature", 0),
            "max_tokens": self.config.get("max_tokens", 800),
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": "enabled" if self.config.get("thinking_enabled", False) else "disabled"
            },
        }
        if self.config.get("thinking_enabled", False):
            payload["reasoning_effort"] = self.config.get("reasoning_effort", "high")
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.get("base_url", "https://api.deepseek.com/chat/completions"),
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        max_retries = max(int(self.config.get("max_retries", 3)), 1)
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=float(self.config.get("timeout_seconds", 60))
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                result = normalize_ai_result(
                    extract_json_object(body["choices"][0]["message"]["content"]),
                    source="deepseek",
                )
                self._append_cache(key, result)
                time.sleep(float(self.config.get("request_interval_seconds", 0.2)))
                return result
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                if attempt + 1 < max_retries:
                    time.sleep(2**attempt)
        fallback["analysis_source"] = "rule_after_ai_failure"
        return fallback

    def _append_cache(self, key: str, result: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"text_hash": key, "result": result}, ensure_ascii=False) + "\n")
        self.cache[key] = result


def cronbach_alpha(frame: pd.DataFrame) -> float | None:
    clean = frame.dropna()
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return None
    item_variance = clean.var(axis=0, ddof=1).sum()
    total_variance = clean.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_variance) or total_variance <= 0:
        return None
    k = clean.shape[1]
    return float(k / (k - 1) * (1 - item_variance / total_variance))


def score_grade(score: float) -> str:
    if pd.isna(score):
        return "缺失"
    if score >= 90:
        return "优秀"
    if score >= 80:
        return "良好"
    if score >= 70:
        return "中上"
    if score >= 60:
        return "一般"
    return "较弱"


def deep_merge_config(*sources: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for src in sources:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge_config(result[key], value)
            else:
                result[key] = value
    return result


def load_pipeline_config(config_path: str | None) -> dict[str, Any]:
    base = {
        "scoring": DEFAULT_SCORING,
        "quality": DEFAULT_QUALITY,
        "split": DEFAULT_SPLIT,
        "privacy": DEFAULT_PRIVACY,
        "model_dataset": DEFAULT_MODEL_DATASET,
        "ai": DEFAULT_AI,
    }
    if not config_path:
        return base
    override = load_json(Path(config_path))
    return deep_merge_config(base, override)


def build_processed_frame(
    raw: pd.DataFrame,
    config: dict[str, Any],
    enable_ai: bool,
    cache_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    question_columns, text_column, metadata = detect_columns(raw)
    missing_questions = [number for number in range(1, QUESTION_COUNT + 1) if number not in question_columns]
    if missing_questions:
        raise ValueError(f"缺少评分题列: {missing_questions}")

    processed = pd.DataFrame(index=raw.index)
    record_column = metadata["record_id"]
    processed["respondent_id"] = (
        raw[record_column].map(canonical_text)
        if record_column
        else pd.Series(np.arange(1, len(raw) + 1), index=raw.index).astype(str)
    )
    for number in range(1, QUESTION_COUNT + 1):
        processed[f"q{number}"] = raw[question_columns[number]].map(normalize_likert)

    input_value_order = config["scoring"].get("input_value_order", "reverse_order")
    if input_value_order not in {"direct", "reverse_order"}:
        raise ValueError("scoring.input_value_order 只能是 direct 或 reverse_order")
    if input_value_order == "reverse_order":
        for number in range(1, QUESTION_COUNT + 1):
            column = f"q{number}"
            processed[column] = processed[column].map(
                lambda value: 6.0 - value if pd.notna(value) else np.nan
            )

    placeholders = set(config["quality"].get("placeholder_texts", []))
    feedback_pairs = (
        raw[text_column].map(lambda value: normalize_feedback(value, placeholders))
        if text_column
        else pd.Series([("无明显不适", False)] * len(raw), index=raw.index)
    )
    processed["review_text"] = feedback_pairs.map(lambda item: item[0])
    processed["has_meaningful_feedback"] = feedback_pairs.map(lambda item: item[1])

    for output_name, metadata_key in (
        ("submitted_at", "submitted_at"),
        ("source_type", "source"),
        ("source_detail", "source_detail"),
    ):
        source_column = metadata[metadata_key]
        processed[output_name] = raw[source_column].map(canonical_text) if source_column else ""

    duration_column = metadata["duration"]
    if duration_column == "duration_seconds":
        processed["duration_seconds"] = pd.to_numeric(raw[duration_column], errors="coerce")
    elif duration_column:
        processed["duration_seconds"] = raw[duration_column].map(parse_duration_seconds)
    else:
        processed["duration_seconds"] = np.nan

    ip_column = metadata["ip"]
    raw_ip = raw[ip_column].map(canonical_text) if ip_column else pd.Series("", index=raw.index)
    processed["ip_group_hash"] = raw_ip.map(lambda value: text_hash(value)[:16] if value else "")
    if config.get("privacy", {}).get("retain_raw_ip", False):
        processed["raw_ip"] = raw_ip

    dimensions = config["scoring"]["dimensions"]
    for name, dimension in dimensions.items():
        columns = [f"q{number}" for number in dimension["items"]]
        processed[name] = processed[columns].mean(axis=1, skipna=False) * 20.0
    processed["target_score"] = sum(
        processed[name] * float(dimension["weight"]) for name, dimension in dimensions.items()
    )
    processed["score_grade"] = processed["target_score"].map(score_grade)
    processed["valid_items"] = processed[[f"q{i}" for i in range(1, 21)]].notna().sum(axis=1)

    q_columns = [f"q{i}" for i in range(1, 21)]
    q_frame = processed[q_columns]
    processed["answer_pattern_hash"] = q_frame.apply(
        lambda row: text_hash("|".join("NA" if pd.isna(value) else str(int(value)) for value in row)),
        axis=1,
    )
    processed["answer_pattern_count"] = processed.groupby("answer_pattern_hash")[
        "answer_pattern_hash"
    ].transform("size")
    processed["ip_submission_count"] = (
        processed.groupby("ip_group_hash")["ip_group_hash"].transform("size")
        if ip_column
        else 1
    )

    quality = config["quality"]
    processed["flag_missing_likert"] = processed["valid_items"] < QUESTION_COUNT
    processed["flag_short_duration"] = (
        processed["duration_seconds"].notna()
        & (processed["duration_seconds"] < float(quality["min_duration_seconds"]))
    )
    processed["flag_straight_line"] = q_frame.nunique(axis=1, dropna=True) == 1
    processed["flag_low_personal_variance"] = (
        q_frame.std(axis=1, skipna=True) < float(quality["low_personal_std_threshold"])
    )
    processed["flag_duplicate_answer_pattern"] = processed["answer_pattern_count"] > 1
    processed["flag_duplicate_ip"] = processed["ip_submission_count"] > 1
    prior_mean = q_frame[[f"q{i}" for i in range(1, 20)]].mean(axis=1)
    processed["flag_overall_inconsistency"] = (
        (processed["q20"] - prior_mean).abs()
        > float(quality["overall_inconsistency_threshold"])
    )

    penalty_map = {
        "flag_short_duration": "short_duration",
        "flag_straight_line": "straight_line",
        "flag_low_personal_variance": "low_personal_variance",
        "flag_duplicate_answer_pattern": "duplicate_answer_pattern",
        "flag_duplicate_ip": "duplicate_ip",
        "flag_overall_inconsistency": "overall_inconsistency",
    }
    quality_score = pd.Series(100.0, index=processed.index)
    for flag_column, penalty_name in penalty_map.items():
        quality_score -= processed[flag_column].astype(float) * float(
            quality["penalties"].get(penalty_name, 0)
        )
    processed["quality_score"] = quality_score.clip(lower=0, upper=100)
    hard_invalid = processed["flag_missing_likert"]
    processed["is_valid"] = (~hard_invalid) & (
        processed["quality_score"] >= float(quality["valid_score_threshold"])
    )
    processed["quality_status"] = np.select(
        [
            hard_invalid,
            processed["quality_score"] < float(quality["valid_score_threshold"]),
            processed["quality_score"] < float(quality["review_score_threshold"]),
        ],
        ["invalid", "review", "review"],
        default="pass",
    )
    flag_columns = ["flag_missing_likert", *penalty_map.keys()]
    processed["quality_flags"] = processed[flag_columns].apply(
        lambda row: ";".join(column.removeprefix("flag_") for column, value in row.items() if value),
        axis=1,
    )

    analyzer = DeepSeekAnalyzer(config["ai"], cache_path=cache_path, enabled=enable_ai)
    unique_feedback: dict[tuple[str, bool], dict[str, Any]] = {}
    for text, meaningful in processed[["review_text", "has_meaningful_feedback"]].itertuples(
        index=False, name=None
    ):
        key = (text, bool(meaningful))
        if key not in unique_feedback:
            unique_feedback[key] = analyzer.analyze(text, bool(meaningful))
    analyses = [
        unique_feedback[(text, bool(meaningful))]
        for text, meaningful in processed[["review_text", "has_meaningful_feedback"]].itertuples(
            index=False, name=None
        )
    ]
    processed["ai_valid_feedback"] = [coerce_bool(item.get("valid_feedback", False)) for item in analyses]
    processed["ai_has_discomfort"] = [coerce_bool(item.get("has_discomfort", False)) for item in analyses]
    processed["ai_sentiment"] = [canonical_text(item.get("sentiment", "neutral")) for item in analyses]
    processed["ai_sentiment_score"] = processed["ai_sentiment"].map(
        {"negative": -1, "neutral": 0, "positive": 1, "no_feedback": 0}
    ).fillna(0)
    processed["ai_problem_types"] = [
        "|".join(coerce_string_list(item.get("problem_types", []))) for item in analyses
    ]
    processed["ai_scene"] = [canonical_text(item.get("scene", "")) for item in analyses]
    processed["ai_symptoms"] = [
        "|".join(coerce_string_list(item.get("symptoms", []))) for item in analyses
    ]
    processed["ai_suggestion"] = [canonical_text(item.get("suggestion", "")) for item in analyses]
    processed["ai_summary"] = [canonical_text(item.get("summary", "")) for item in analyses]
    processed["ai_confidence"] = [float(item.get("confidence", 0.0) or 0.0) for item in analyses]
    processed["ai_analysis_source"] = [
        canonical_text(item.get("analysis_source", "rule")) for item in analyses
    ]

    mapping = {
        "questions": {f"q{number}": question_columns[number] for number in range(1, 21)},
        "review_text": text_column,
        "metadata": metadata,
        "dimensions": dimensions,
    }
    return processed, mapping


def assign_splits(processed: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = processed.copy()
    result["dataset_split"] = "review"
    valid_index = result.index[result["is_valid"]]
    if len(valid_index) < 2:
        return result
    test_ratio = float(config["split"].get("test_ratio", 0.2))
    seed = int(config["split"].get("random_seed", 42))
    rng = np.random.default_rng(seed)
    grades = result.loc[valid_index, "score_grade"]
    test_parts: list[int] = []
    if grades.value_counts().min() >= 2 and grades.nunique() > 1:
        for _, group in grades.groupby(grades):
            indices = group.index.to_numpy(copy=True)
            rng.shuffle(indices)
            group_test_size = max(1, min(len(indices) - 1, int(round(len(indices) * test_ratio))))
            test_parts.extend(indices[:group_test_size].tolist())
        test_index = pd.Index(test_parts)
        train_index = valid_index.difference(test_index)
    else:
        shuffled = valid_index.to_numpy(copy=True)
        rng.shuffle(shuffled)
        test_size = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_ratio))))
        test_index = pd.Index(shuffled[:test_size])
        train_index = pd.Index(shuffled[test_size:])
    result.loc[train_index, "dataset_split"] = "train"
    result.loc[test_index, "dataset_split"] = "test"
    return result


def build_model_dataset(processed: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "review_text",
        "duration_seconds",
        "has_meaningful_feedback",
        "ai_has_discomfort",
        "ai_sentiment_score",
        "ai_confidence",
        "target_score",
        "dataset_split",
    ]
    model_config = config.get("model_dataset", {})
    if model_config.get("include_question_items", False):
        columns[1:1] = [f"q{i}" for i in range(1, 21)]
    if model_config.get("include_derived_dimensions", False):
        columns[1:1] = list(config["scoring"]["dimensions"].keys())
    final_columns = [c for c in columns if c in processed.columns]
    model = processed[final_columns].copy()
    model.insert(1, "image_path", "PLACEHOLDER_IMAGE")
    model.insert(2, "audio_path", "PLACEHOLDER_AUDIO")
    numeric_columns = model.select_dtypes(include="number").columns
    model[numeric_columns] = model[numeric_columns].fillna(0)
    model["has_meaningful_feedback"] = model["has_meaningful_feedback"].astype(int)
    model["ai_has_discomfort"] = model["ai_has_discomfort"].astype(int)
    return model


def build_report(
    processed: pd.DataFrame,
    config: dict[str, Any],
    input_path: Path,
    column_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dimensions = config["scoring"]["dimensions"]
    q_columns = [f"q{i}" for i in range(1, 21)]
    quality_flag_columns = [column for column in processed.columns if column.startswith("flag_")]
    report: dict[str, Any] = {
        "analysis_time": datetime.now().isoformat(),
        "input_file": str(input_path.resolve()),
        "scoring_version": "20-item-5-per-dimension-v1",
        "input_value_order": config["scoring"].get("input_value_order", "reverse_order"),
        "total_records": int(len(processed)),
        "valid_records": int(processed["is_valid"].sum()),
        "review_records": int((processed["quality_status"] == "review").sum()),
        "invalid_records": int((processed["quality_status"] == "invalid").sum()),
        "split_counts": {str(k): int(v) for k, v in processed["dataset_split"].value_counts().items()},
        "target_score": {
            "mean": round(float(processed["target_score"].mean()), 4),
            "std": round(float(processed["target_score"].std()), 4),
            "min": round(float(processed["target_score"].min()), 4),
            "median": round(float(processed["target_score"].median()), 4),
            "max": round(float(processed["target_score"].max()), 4),
        },
        "dimensions": {},
        "grades": {str(k): int(v) for k, v in processed["score_grade"].value_counts().items()},
        "quality_flags": {column: int(processed[column].sum()) for column in quality_flag_columns},
        "open_text": {
            "meaningful_records": int(processed["has_meaningful_feedback"].sum()),
            "unique_texts": int(processed["review_text"].nunique()),
            "deepseek_analyzed_records": int(
                (processed["ai_analysis_source"] == "deepseek").sum()
            ),
            "problem_type_counts": dict(
                Counter(
                    problem
                    for value in processed["ai_problem_types"]
                    for problem in str(value).split("|")
                    if problem
                )
            ),
        },
        "ai_configuration": {
            "provider": config["ai"].get("provider", "deepseek"),
            "model": config["ai"].get("model", "deepseek-v4-flash"),
            "thinking_enabled": bool(config["ai"].get("thinking_enabled", False)),
            "reasoning_effort": config["ai"].get("reasoning_effort", "none"),
            "response_format": "json_object",
        },
        "reliability": {
            "cronbach_alpha_total": cronbach_alpha(processed[q_columns]),
        },
        "method_notes": [
            "四个一级维度均按5个题项计算，题项均为1-5分正向计分。",
            "input_value_order=direct 表示导出值就是实际分数；reverse_order 表示问卷选项按“5、4、3、2、1”顺序排列，导出值为选项位置，按 实际分数 = 6 - 导出值 换算。",
            "质量标记用于筛查与人工复核，不等同于对受访者真实性作最终认定。",
            "DeepSeek 仅接收脱敏后的开放题文本，不接收 IP、时间或问卷编号。",
            "默认模型数据不包含 q1-q20 和四个派生维度，避免用生成标签的字段预测同一标签。",
        ],
    }
    for name, dimension in dimensions.items():
        columns = [f"q{i}" for i in dimension["items"]]
        report["dimensions"][name] = {
            "name_cn": dimension["name_cn"],
            "items": columns,
            "weight": dimension["weight"],
            "mean": round(float(processed[name].mean()), 4),
            "std": round(float(processed[name].std()), 4),
            "cronbach_alpha": cronbach_alpha(processed[columns]),
        }
    return report


def normalize_legacy_column_mapping(legacy: dict[str, Any]) -> dict[str, Any]:
    if "questions" in legacy:
        return legacy
    q_mapping: dict[str, str] = {}
    for key, value in legacy.items():
        if isinstance(key, str) and re.fullmatch(r"q\d{1,2}", key):
            q_mapping[key] = str(value)
    return {
        "questions": q_mapping,
        "review_text": None,
        "metadata": {
            "record_id": None,
            "submitted_at": None,
            "duration": None,
            "source": None,
            "source_detail": None,
            "ip": None,
        },
        "dimensions": DEFAULT_SCORING["dimensions"],
        "legacy_mapping": True,
    }


def process_questionnaire(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    enable_ai: bool = False,
) -> dict[str, Any]:
    config = load_pipeline_config(str(config_path) if config_path else None)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(input_path)
    processed, mapping = build_processed_frame(
        raw,
        config,
        enable_ai=enable_ai,
        cache_path=output_dir / "ai_cache.jsonl",
    )
    processed = assign_splits(processed, config)
    model_dataset = build_model_dataset(processed, config)
    report = build_report(processed, config, input_path, column_mapping=mapping)

    processed.to_csv(output_dir / "questionnaire_analysis.csv", index=False, encoding="utf-8-sig")
    model_dataset.to_csv(output_dir / "aict_dataset.csv", index=False, encoding="utf-8-sig")
    model_dataset[model_dataset["dataset_split"] == "train"].to_csv(
        output_dir / "train_dataset.csv", index=False, encoding="utf-8-sig"
    )
    model_dataset[model_dataset["dataset_split"] == "test"].to_csv(
        output_dir / "test_dataset.csv", index=False, encoding="utf-8-sig"
    )
    processed[processed["quality_status"] != "pass"].to_csv(
        output_dir / "review_queue.csv", index=False, encoding="utf-8-sig"
    )
    write_json(output_dir / "column_mapping.json", mapping)
    write_json(output_dir / "analysis_report.json", report)

    print(
        json.dumps(
            {
                "input_records": len(raw),
                "valid_records": int(processed["is_valid"].sum()),
                "review_records": int((processed["quality_status"] == "review").sum()),
                "output_dir": str(output_dir.resolve()),
                "ai_requested": bool(enable_ai),
                "deepseek_records": int(
                    (processed["ai_analysis_source"] == "deepseek").sum()
                ),
            },
            ensure_ascii=False,
        )
    )
    return {
        "processed": processed,
        "model_dataset": model_dataset,
        "mapping": mapping,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="处理AI+文旅20题问卷并生成AICT数据集")
    parser.add_argument("--input", required=True, help="问卷星原始CSV或现有scored_full.csv")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "questionnaire_pipeline.json"),
        help="处理配置 JSON",
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="调用DeepSeek分析有效开放题；需要设置DEEPSEEK_API_KEY",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(Path(args.config))
    raw = read_csv(input_path)
    processed, mapping = build_processed_frame(
        raw,
        config,
        enable_ai=args.enable_ai,
        cache_path=output_dir / "ai_cache.jsonl",
    )
    processed = assign_splits(processed, config)
    model_dataset = build_model_dataset(processed, config)
    report = build_report(processed, config, input_path)

    processed.to_csv(output_dir / "questionnaire_analysis.csv", index=False, encoding="utf-8-sig")
    model_dataset.to_csv(output_dir / "aict_dataset.csv", index=False, encoding="utf-8-sig")
    model_dataset[model_dataset["dataset_split"] == "train"].to_csv(
        output_dir / "train_dataset.csv", index=False, encoding="utf-8-sig"
    )
    model_dataset[model_dataset["dataset_split"] == "test"].to_csv(
        output_dir / "test_dataset.csv", index=False, encoding="utf-8-sig"
    )
    processed[processed["quality_status"] != "pass"].to_csv(
        output_dir / "review_queue.csv", index=False, encoding="utf-8-sig"
    )
    write_json(output_dir / "column_mapping.json", mapping)
    write_json(output_dir / "analysis_report.json", report)
    print(
        json.dumps(
            {
                "input_records": len(raw),
                "valid_records": int(processed["is_valid"].sum()),
                "review_records": int((processed["quality_status"] == "review").sum()),
                "output_dir": str(output_dir.resolve()),
                "ai_requested": bool(args.enable_ai),
                "deepseek_records": int(
                    (processed["ai_analysis_source"] == "deepseek").sum()
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
