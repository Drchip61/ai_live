"""
表情动作语义映射器

将 LLM 输出的自由文本 #[action][emotion][voice_emotion] 标签映射到
expression_motion_mapping.json 中的固定集合。

采用 sentence embedding + cosine similarity 实现语义最近邻匹配，
预计算目标集合的向量，运行时仅需 embed 查询词并做点积，延迟极低。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\](?:\[([^\]]*)\])?")
_DEFAULT_VOICE_EMOTION = "neutral"

# 中英双语关键词扩展，提升跨语言 embedding 匹配准确率。
# key = mapping.json 中的 name，value = 追加的英文同义词列表。
_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": ["happy", "joyful", "cheerful", "delighted", "sparkle", "starry eyes", "excited", "love", "开心", "快乐"],
    "angry": ["angry", "mad", "annoyed", "frustrated", "furious", "rage", "dark face", "生气", "愤怒"],
    "embarrased": ["embarrassed", "shy", "blush", "flustered", "tsundere", "nervous", "bashful", "脸红", "害羞"],
    "disgust": ["disgust", "cringe", "speechless", "unamused", "bored", "disdain", "嫌弃", "无语"],
    "neutral": ["neutral", "calm", "deadpan", "blank", "poker face", "indifferent", "relaxed", "平静"],
    "sad": ["sad", "cry", "upset", "disappointed", "tearful", "sobbing", "down", "难过", "委屈"],
    "surprised": ["surprised", "shocked", "startled", "mind blown", "overwhelmed", "惊讶", "意外"],
    "wonder": ["wonder", "curious", "interested", "intrigued", "amazed", "thinking", "好奇", "期待"],
}

_VOICE_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": ["happy", "joyful", "cheerful", "delighted", "playful", "开心", "快乐", "雀跃"],
    "angry": ["angry", "mad", "furious", "rage", "irritated", "生气", "火大", "发怒"],
    "embarrased": ["embarrassed", "shy", "bashful", "flustered", "害羞", "不好意思", "扭捏"],
    "disgust": ["disgust", "grossed out", "repulsed", "revolted", "嫌弃", "厌恶", "反感"],
    "neutral": ["neutral", "calm", "gentle", "steady", "serene", "平静", "温柔", "自然"],
    "sad": ["sad", "down", "depressed", "heartbroken", "难过", "失落", "委屈"],
    "surprised": ["surprised", "shocked", "startled", "amazed", "惊讶", "意外", "吃惊"],
    "wonder": ["wonder", "curious", "interested", "intrigued", "anticipation", "好奇", "期待", "感兴趣"],
}

_MOTION_KEYWORDS: dict[str, list[str]] = {
    "idle": ["idle", "stand", "still", "default", "waiting", "calm", "relaxed"],
    "acting_cute": ["cute", "adorable", "coy", "kawaii", "charming", "flirty"],
    "affirm": ["affirm", "approve", "agree", "confirm", "yes", "certainly"],
    "arms_crossed": ["arms crossed", "cross arms", "defensive", "serious", "skeptical"],
    "arms_open": ["arms open", "open arms", "welcome", "embrace", "hug"],
    "cheek_rest": ["cheek rest", "rest cheek", "daydream", "bored", "zoning out"],
    "chin_pinch": ["chin pinch", "pinch chin", "pondering", "deliberate"],
    "chin_rest": ["chin rest", "rest chin", "thinking", "contemplating", "listening"],
    "clap": ["clap", "applause", "bravo", "well done", "praise"],
    "cold": ["cold", "shiver", "chilly", "freezing", "brr"],
    "dance": ["dance", "sway", "groove", "rhythm", "happy dance", "celebrate"],
    "disdain": ["disdain", "contempt", "scorn", "sneer", "disgusted", "look down"],
    "eye_roll": ["eye roll", "roll eyes", "whatever", "unimpressed", "sarcastic"],
    "eye_rub": ["eye rub", "rub eyes", "sleepy", "tired", "disbelief"],
    "face_rest": ["face rest", "cup face", "shy", "bashful"],
    "finger_on_chin": ["finger on chin", "curious", "wondering", "hmm"],
    "fists_up": ["fists up", "pump fist", "fighting", "determined", "lets go"],
    "glance_down": ["glance down", "look down", "shy", "guilty", "nervous"],
    "half_squat": ["squat", "crouch", "playful squat", "ready"],
    "hand_on_chin": ["hand on chin", "deep thought", "analyze", "serious thinking"],
    "hands_behind_back": ["hands behind back", "obedient", "polite", "waiting"],
    "hands_cover_face": ["cover face", "hide face", "embarrassed", "too shy"],
    "hands_on_chin": ["hands on chin", "both hands chin", "attentive", "focused"],
    "hands_on_hips": ["hands on hips", "akimbo", "defiant", "confident", "scolding"],
    "hands_raise": ["hands raise", "surrender", "give up", "helpless"],
    "hands_up": ["hands up", "cheer", "hooray", "excited", "celebrate"],
    "head_shake": ["head shake", "shake head", "no", "deny", "disagree", "refuse"],
    "head_tilt": ["head tilt", "tilt head", "confused", "curious", "cute tilt"],
    "leg_raise": ["leg raise", "kick", "playful kick", "energetic"],
    "look_around": ["look around", "glance around", "alert", "searching", "wary"],
    "look_left_panic": ["panic", "startled", "alarmed", "scared look"],
    "look_right": ["look away", "avert eyes", "avoid", "evasive"],
    "nod": ["nod", "agree", "acknowledge", "yes", "understand", "listen"],
    "peace_sign": ["peace sign", "v sign", "victory", "selfie pose", "yeah"],
    "point_camera": ["point camera", "point at you", "you", "emphasize", "accuse"],
    "pointing": ["point", "indicate", "direct", "show", "guide attention"],
    "praying": ["pray", "please", "beg", "grateful", "thankful", "palms together"],
    "shrugging": ["shrug", "dunno", "whatever", "helpless", "indifferent"],
    "shush": ["shush", "quiet", "silence", "hush", "secret", "whisper"],
    "stop": ["stop", "halt", "wait", "hold on", "enough"],
    "stretch": ["stretch", "yawn", "tired", "relax", "wake up"],
    "thinking": ["think", "ponder", "consider", "reflect", "hmm", "deliberate"],
    "wave": ["wave", "hello", "goodbye", "greeting", "hi", "bye", "welcome"],
}


# 英文 → 中文语义增强，帮助中文 embedding 模型理解英文查询
_EN_ZH_EMOTION: dict[str, str] = {
    "happy": "开心 快乐 高兴 眼前一亮",
    "excited": "兴奋 激动 期待 眼前一亮",
    "friendly": "友好 温暖 亲切 关心",
    "thoughtful": "沉思 认真 思考 分析 严肃 讲道理",
    "sad": "难过 悲伤 伤心 委屈",
    "angry": "生气 愤怒 恼火 发火",
    "sorry": "抱歉 道歉 委屈",
    "shy": "害羞 脸红 不好意思 被拆穿",
    "embarrassed": "尴尬 害臊 脸红 被拆穿",
    "proud": "得意 骄傲 自豪 嘚瑟 调皮 吐舌头",
    "curious": "好奇 感兴趣 想知道 眼前一亮",
    "bored": "无聊 无语 嫌弃 懒得理",
    "annoyed": "烦躁 不耐烦 嫌弃 无语 懒得理",
    "surprised": "惊讶 意外 吃惊 眼前一亮",
    "confused": "困惑 迷茫 不明白 无话可说",
    "determined": "坚定 认真 不服输 好胜",
    "playful": "调皮 搞怪 玩闹 吐舌头 得意",
    "smug": "得意 嘚瑟 沾沾自喜 调皮 吐舌头",
    "nervous": "紧张 忐忑 慌张 害羞 脸红",
    "touched": "感动 真情流露 比心",
    "love": "喜欢 心动 比心 真情流露",
    "worry": "担心 担忧 忧虑 关心",
    "flustered": "手足无措 慌乱 脸红 害羞",
    "cold": "冷淡 高冷 不想理你 嫌弃 无语",
    "speechless": "无语 无话可说 尴尬 脸黑 翻车",
}

_EN_ZH_MOTION: dict[str, str] = {
    "idle": "待机 放松 站立 安静 等待",
    "acting cute": "撒娇 卖萌 可爱 扭动",
    "cute": "撒娇 卖萌 可爱",
    "affirm": "肯定 赞同 确认 用力点头",
    "arms crossed": "抱胸 交叉 防备 冷静",
    "arms open": "张开 欢迎 拥抱 敞开",
    "cheek rest": "托腮 发呆 若有所思",
    "chin pinch": "捏下巴 故作深沉",
    "chin rest": "托下巴 沉思 认真",
    "clap": "鼓掌 拍手 赞赏 好棒",
    "cold": "发冷 打冷战 缩着",
    "dance": "跳舞 摇摆 律动 开心跳",
    "disdain": "嫌弃 不屑 傲慢 瞧不起",
    "eye roll": "翻白眼 无语 受不了",
    "eye rub": "揉眼睛 困倦 不敢相信",
    "face rest": "捧脸 害羞 托腮",
    "finger on chin": "手指抵下巴 好奇 犹豫",
    "fists up": "握拳 加油 干劲",
    "glance down": "低头 害羞 心虚 垂眼",
    "half squat": "半蹲 蓄力 调皮",
    "hand on chin": "撑下巴 深度思考",
    "hands behind back": "背手 乖巧 等待",
    "hands cover face": "捂脸 害羞到不行",
    "hands on chin": "双手托下巴 认真 专注",
    "hands on hips": "叉腰 不服气 气鼓鼓 理直气壮",
    "hands raise": "举手 投降 无奈",
    "hands up": "高举 欢呼 庆祝",
    "head shake": "摇头 否认 不同意 拒绝",
    "shake head": "摇头 否认 不同意",
    "head tilt": "歪头 疑惑 可爱",
    "leg raise": "抬腿 活泼 踢一下",
    "look around": "东张西望 环顾 警觉",
    "panic": "惊慌 心虚 被吓到",
    "look away": "看别处 回避 移开视线",
    "nod": "点头 同意 理解 嗯嗯",
    "peace sign": "比耶 剪刀手 胜利",
    "point": "指向镜头 指着你 强调",
    "pointing": "指向 引导 注意力",
    "pray": "双手合十 拜托 祈求 感恩",
    "praying": "双手合十 拜托 祈求",
    "shrug": "耸肩 无所谓 我也没办法",
    "shush": "嘘 安静 小声点 保密",
    "stop": "制止 住手 等一下",
    "stretch": "伸懒腰 放松 累了 打哈欠",
    "thinking": "思考 认真想 沉思",
    "wave": "挥手 打招呼 欢迎 再见",
    "laugh": "跳舞 大笑 开心",
    "smile": "待机 微笑 温和 平静",
    "cheering up": "欢呼 高举 庆祝 兴奋",
    "ok gesture": "点头 认同 肯定",
    "heart gesture": "双手合十 祈祷 真情",
    "sigh": "举手 投降 无奈 叹气",
    "stomp": "叉腰 生气 跺脚 不服",
    "spin": "跳舞 转圈 得意",
    "jump": "欢呼 跳跃 兴奋 庆祝",
}

_EN_ZH_VOICE_EMOTION: dict[str, str] = {
    "joy": "喜悦 开心 快乐 雀跃 轻快",
    "anticipation": "期待 兴奋 激动 跃跃欲试 迫不及待",
    "anger": "生气 愤怒 火大 恼火 发火",
    "disgust": "嫌弃 厌恶 反感 无语 不耐烦",
    "sadness": "难过 失落 悲伤 委屈 沮丧",
    "surprise": "惊讶 意外 吃惊 吓一跳",
    "fear": "害怕 紧张 担心 惊恐 忐忑",
    "serenity": "平静 温柔 放松 稳定 自然",
    "curiosity": "好奇 想知道 感兴趣 探究",
    "agitation": "烦躁 焦躁 急了 情绪上来 不安",
    "shyness": "害羞 不好意思 扭捏 脸红",
    "indignation": "不服 愤愤不平 抗议 觉得委屈 被冤枉",
    "happy": "喜悦 开心 快乐 雀跃",
    "excited": "期待 兴奋 激动 跃跃欲试",
    "friendly": "平静 温柔 亲切 自然",
    "thoughtful": "平静 稳定 冷静 温和",
    "sad": "难过 失落 悲伤 委屈",
    "angry": "生气 愤怒 恼火 发火",
    "sorry": "难过 低落 委屈 抱歉",
    "shy": "害羞 不好意思 扭捏 脸红",
    "embarrassed": "害羞 尴尬 不好意思 慌乱",
    "proud": "喜悦 得意 开心 自信",
    "curious": "好奇 想知道 感兴趣",
    "bored": "平静 冷淡 无聊 放空",
    "annoyed": "烦躁 焦躁 不耐烦 火气上来",
    "surprised": "惊讶 意外 吃惊 吓一跳",
    "confused": "好奇 困惑 想弄明白",
    "determined": "期待 坚定 蓄势待发 想冲",
    "playful": "喜悦 轻快 调皮 活泼",
    "smug": "不服 得意 抗议 嘚瑟",
    "nervous": "害怕 紧张 忐忑",
    "touched": "平静 温柔 感动 安静",
    "love": "喜悦 温柔 真诚 开心",
    "worry": "害怕 担心 忧虑",
    "flustered": "害羞 慌乱 手足无措",
    "cold": "嫌弃 冷淡 反感",
    "speechless": "嫌弃 无语 反感",
}


@dataclass
class MappedTag:
    """一个 #[action][emotion][voice_emotion] 标签的映射结果"""
    original_action: str
    original_emotion: str
    original_voice_emotion: str
    mapped_motion: str
    mapped_expression: str
    mapped_voice_emotion: str
    motion_score: float
    expression_score: float
    voice_emotion_score: float


@dataclass
class MapperResult:
    """整句文本的映射结果"""
    mapped_text: str
    tags: list[MappedTag] = field(default_factory=list)


class ExpressionMotionMapper:
    """
    语义映射器：LLM 自由标签 → 固定动作/表情/语音情绪集

    初始化时从 mapping JSON 加载目标集合并预计算 embeddings，
    运行时用 cosine similarity 做最近邻查找。
    查询结果会缓存，相同词不会重复 embed。
    """

    def __init__(
        self,
        mapping_path: Path,
        embeddings=None,
    ) -> None:
        self._cache: dict[str, tuple[str, float]] = {}

        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        self._emotions: list[dict] = raw.get("emotions", [])
        self._voice_emotions: list[dict] = raw.get("voice_emotions", [])
        self._motions: list[dict] = raw.get("motions", [])

        self._embeddings = embeddings
        self._emotion_names: list[str] = [e["name"] for e in self._emotions]
        self._voice_emotion_names: list[str] = [v["name"] for v in self._voice_emotions]
        self._motion_names: list[str] = [m["name"] for m in self._motions]

        self._emotion_vectors: Optional[np.ndarray] = None
        self._voice_emotion_vectors: Optional[np.ndarray] = None
        self._motion_vectors: Optional[np.ndarray] = None

        if embeddings is not None:
            self._precompute()

    def lazy_init(self, embeddings) -> None:
        """延迟初始化 embedding 模型（等 MemoryManager 就绪后调用）"""
        if self._emotion_vectors is not None:
            return
        self._embeddings = embeddings
        self._precompute()

    def _precompute(self) -> None:
        """预计算目标集合的 embedding 向量"""
        emotion_texts = []
        for e in self._emotions:
            kw = _EMOTION_KEYWORDS.get(e["name"], [])
            text = f"{e['name']} ({', '.join(kw)}): {e['desc']}" if kw else f"{e['name']}: {e['desc']}"
            emotion_texts.append(text)

        voice_texts = []
        for v in self._voice_emotions:
            kw = _VOICE_EMOTION_KEYWORDS.get(v["name"], [])
            text = f"{v['name']} ({', '.join(kw)}): {v['desc']}" if kw else f"{v['name']}: {v['desc']}"
            voice_texts.append(text)

        motion_texts = []
        for m in self._motions:
            kw = _MOTION_KEYWORDS.get(m["name"], [])
            text = f"{m['name']} ({', '.join(kw)}): {m['desc']}" if kw else f"{m['name']}: {m['desc']}"
            motion_texts.append(text)

        try:
            ev = self._embeddings.embed_documents(emotion_texts)
            vv = self._embeddings.embed_documents(voice_texts)
            mv = self._embeddings.embed_documents(motion_texts)
            self._emotion_vectors = self._normalize(np.array(ev, dtype=np.float32))
            self._voice_emotion_vectors = self._normalize(np.array(vv, dtype=np.float32))
            self._motion_vectors = self._normalize(np.array(mv, dtype=np.float32))
            logger.info(
                "ExpressionMotionMapper 预计算完成: %d emotions, %d voice_emotions, %d motions",
                len(self._emotion_names), len(self._voice_emotion_names), len(self._motion_names),
            )
        except Exception as e:
            logger.error("ExpressionMotionMapper 预计算失败: %s", e)

    def map_response(self, text: str) -> MapperResult:
        """
        解析并映射整段回复文本中所有标签。

        Returns:
            MapperResult，包含替换后文本和每个标签的映射详情。
        """
        if (
            self._emotion_vectors is None
            or self._voice_emotion_vectors is None
            or self._motion_vectors is None
        ):
            return MapperResult(mapped_text=text)

        tags: list[MappedTag] = []

        def _replace(match: re.Match) -> str:
            action_raw = match.group(1).strip()
            emotion_raw = match.group(2).strip()
            voice_raw = (match.group(3) or "").strip()
            voice_query = voice_raw or emotion_raw or _DEFAULT_VOICE_EMOTION

            mapped_motion, m_score = self._find_nearest_motion(action_raw)
            mapped_expr, e_score = self._find_nearest_emotion(emotion_raw)
            mapped_voice, v_score = self._find_nearest_voice_emotion(voice_query)

            tags.append(MappedTag(
                original_action=action_raw,
                original_emotion=emotion_raw,
                original_voice_emotion=voice_query,
                mapped_motion=mapped_motion,
                mapped_expression=mapped_expr,
                mapped_voice_emotion=mapped_voice,
                motion_score=round(m_score, 4),
                expression_score=round(e_score, 4),
                voice_emotion_score=round(v_score, 4),
            ))
            return f"#[{mapped_motion}][{mapped_expr}][{mapped_voice}]"

        mapped_text = _TAG_PATTERN.sub(_replace, text)
        return MapperResult(mapped_text=mapped_text, tags=tags)

    def _find_nearest_emotion(self, query: str) -> tuple[str, float]:
        cache_key = f"e:{query}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        enriched = self._enrich_query(query, _EN_ZH_EMOTION)
        result = self._nearest(enriched, self._emotion_names, self._emotion_vectors)
        self._cache[cache_key] = result
        return result

    def _find_nearest_motion(self, query: str) -> tuple[str, float]:
        cache_key = f"m:{query}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        enriched = self._enrich_query(query, _EN_ZH_MOTION)
        result = self._nearest(enriched, self._motion_names, self._motion_vectors)
        self._cache[cache_key] = result
        return result

    def _find_nearest_voice_emotion(self, query: str) -> tuple[str, float]:
        cache_key = f"v:{query}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        enriched = self._enrich_query(query, _EN_ZH_VOICE_EMOTION)
        result = self._nearest(enriched, self._voice_emotion_names, self._voice_emotion_vectors)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _enrich_query(query: str, lookup: dict[str, str]) -> str:
        """将英文查询增强为中英双语，提升中文 embedding 模型的匹配准确率"""
        key = query.lower().strip()
        zh = lookup.get(key)
        if zh:
            return f"{query} {zh}"
        for en_key, zh_val in lookup.items():
            if en_key in key or key in en_key:
                return f"{query} {zh_val}"
        return query

    def _nearest(
        self,
        query: str,
        names: list[str],
        vectors: np.ndarray,
    ) -> tuple[str, float]:
        """用 cosine similarity 找最近邻"""
        try:
            qv = np.array(
                self._embeddings.embed_query(query), dtype=np.float32,
            ).reshape(1, -1)
            qv = self._normalize(qv)
            scores = (qv @ vectors.T).flatten()
            idx = int(np.argmax(scores))
            return names[idx], float(scores[idx])
        except Exception as e:
            logger.error("语义匹配失败 (query=%s): %s", query, e)
            return names[0], 0.0

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return v / norms
