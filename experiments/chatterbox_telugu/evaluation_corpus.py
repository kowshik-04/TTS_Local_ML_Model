"""Fixed, categorized Telugu and Telugu-English evaluation prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "references"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    text: str
    language_id: str
    reference_id: str = "01_conversation"


PURE_TELUGU = (
    EvaluationCase("te001", "conversation", "ఈ రోజు ఉదయం వాతావరణం చాలా ఆహ్లాదంగా ఉంది.", "te"),
    EvaluationCase("te002", "conversation", "మీరు భోజనం చేశారా? నేను ఇప్పుడే చేశాను.", "te"),
    EvaluationCase("te003", "conversation", "సాయంత్రం మనం కలిసి నడకకు వెళ్దామా?", "te"),
    EvaluationCase("te004", "conversation", "నాకు ఒక కప్పు వేడి కాఫీ కావాలి.", "te"),
    EvaluationCase("te005", "conversation", "రేపు మీకు సమయం ఉంటే నాకు చెప్పండి.", "te"),
    EvaluationCase("te006", "formal", "దయచేసి ఈ పత్రాన్ని జాగ్రత్తగా పరిశీలించండి.", "te"),
    EvaluationCase("te007", "formal", "మీ సహకారానికి హృదయపూర్వక ధన్యవాదాలు.", "te"),
    EvaluationCase("te008", "formal", "సమావేశం ప్రారంభమయ్యే ముందు అందరూ తమ స్థానాల్లో కూర్చోవాలి.", "te"),
    EvaluationCase("te009", "vowels_aspiration", "ఆకాశంలో ఎగురుతున్న పక్షులను పిల్లలు ఆసక్తిగా చూశారు.", "te"),
    EvaluationCase("te010", "vowels_aspiration", "ఖాళీ పాత్రను శుభ్రంగా కడిగి అలమారలో పెట్టండి.", "te"),
    EvaluationCase("te011", "vowels_aspiration", "భారీ వర్షం పడినా రైతులు పొలానికి వెళ్లారు.", "te"),
    EvaluationCase("te012", "vowels_aspiration", "దీర్ఘకాలిక ఆరోగ్యానికి ప్రశాంతమైన నిద్ర అవసరం.", "te"),
    EvaluationCase("te013", "conjuncts", "స్వాతంత్ర్యం ప్రతి పౌరుడి బాధ్యతను గుర్తు చేస్తుంది.", "te"),
    EvaluationCase("te014", "conjuncts", "జ్ఞానం మరియు విజ్ఞానం సమాజ అభివృద్ధికి తోడ్పడతాయి.", "te"),
    EvaluationCase("te015", "conjuncts", "కృష్ణుడు క్షమతో సమస్యను పరిష్కరించాడు.", "te"),
    EvaluationCase("te016", "conjuncts", "పుస్తకంలోని ప్రశ్నలకు విద్యార్థులు సమాధానాలు రాశారు.", "te"),
    EvaluationCase("te017", "similar_sounding_words", "పాలు తీసుకుని బాలుడికి ఇచ్చి, పక్కనే ఉన్న పలకను శుభ్రం చేశాను.", "te"),
    EvaluationCase("te018", "similar_sounding_words", "కళాకారుడు కలతో కొత్త చిత్రాన్ని గీశాడు.", "te"),
    EvaluationCase("te019", "similar_sounding_words", "చిరు జల్లులో చిన్నారి చిరునవ్వుతో ఆడుకుంది.", "te"),
    EvaluationCase("te020", "similar_sounding_words", "నది ఒడ్డున నిలబడి నీటి అలలను చూశాము.", "te"),
    EvaluationCase("te021", "punctuation", "అమ్మా, మీరు ఎప్పుడు వస్తారు? మేము ఎదురుచూస్తున్నాము.", "te"),
    EvaluationCase("te022", "punctuation", "అద్భుతం! ఆ చిన్నారి ఎంత చక్కగా పాడింది.", "te"),
    EvaluationCase("te023", "punctuation", "అతను అన్నాడు: “నిజాయితీ ఎప్పుడూ గెలుస్తుంది.”", "te"),
    EvaluationCase("te024", "numbers", "రేపు ఉదయం 10 గంటలకు సమావేశం ప్రారంభమవుతుంది.", "te"),
    EvaluationCase("te025", "numbers", "ఈ పుస్తకానికి 250 రూపాయలు చెల్లించాను.", "te"),
    EvaluationCase("te026", "numbers", "మా గ్రామం నగరానికి దాదాపు 18 కిలోమీటర్ల దూరంలో ఉంది.", "te"),
    EvaluationCase("te027", "names", "కౌశిక్ విజయవాడ నుంచి ఢిల్లీకి ప్రయాణించాడు.", "te"),
    EvaluationCase("te028", "names", "సీత, రమేష్ మరియు అనన్య గ్రంథాలయంలో కలుసుకున్నారు.", "te"),
    EvaluationCase("te029", "long_sentence", "మనం ఏదైనా పని ప్రారంభించే ముందు దాని గురించి పూర్తిగా అర్థం చేసుకుని, సరైన విధంగా ప్రణాళిక చేసుకుంటే ఆ పనిని మరింత సమర్థవంతంగా పూర్తి చేయగలం.", "te"),
    EvaluationCase("te030", "long_sentence", "ఉదయం త్వరగా లేచి వ్యాయామం చేసిన తరువాత అల్పాహారం తీసుకుని, రోజంతా చేయాల్సిన పనులను క్రమంగా పూర్తి చేయాలని ఆమె నిర్ణయించుకుంది.", "te"),
)

MIXED_TELUGU_ENGLISH = (
    EvaluationCase("mix001", "mixed_technical", "నేను Linux కంప్యూటర్‌లో కొత్త Python ప్రోగ్రామ్‌ను పరీక్షిస్తున్నాను.", "te", "08_telugu_english"),
    EvaluationCase("mix002", "mixed_technical", "Red Hat బృందం Kubernetes క్లస్టర్‌ను జాగ్రత్తగా నిర్వహిస్తోంది.", "te", "08_telugu_english"),
    EvaluationCase("mix003", "mixed_technical", "ఈ API ద్వారా Cloud సేవకు సురక్షితంగా కనెక్ట్ కావచ్చు.", "te", "08_telugu_english"),
    EvaluationCase("mix004", "mixed_conversation", "ఈ రోజు online meeting ఉందా, లేక రేపటికి మార్చారా?", "te", "08_telugu_english"),
    EvaluationCase("mix005", "mixed_conversation", "నేను message పంపాను, మీరు వచ్చినప్పుడు reply ఇవ్వండి.", "te", "08_telugu_english"),
    EvaluationCase("mix006", "mixed_names", "Ananya హైదరాబాద్‌లోని office నుంచి Zoom callలో చేరింది.", "te", "08_telugu_english"),
    EvaluationCase("mix007", "mixed_technical", "కొత్త software update తర్వాత database backup సరిగ్గా ఉందో చూడండి.", "te", "08_telugu_english"),
    EvaluationCase("mix008", "mixed_long_sentence", "ప్రాజెక్ట్ గడువులోపు పూర్తిచేయడానికి team రోజూ progressను పంచుకుని, అవసరమైనప్పుడు designలో చిన్న మార్పులు చేస్తుంది.", "te", "08_telugu_english"),
)

CASES = PURE_TELUGU + MIXED_TELUGU_ENGLISH

REFERENCE_TRANSCRIPTS = {
    "01_conversation": "నమస్కారం! నా పేరు కౌశిక్. నేను విజయవాడ నుంచి వచ్చాను. ప్రస్తుతం ఢిల్లీలో పనిచేస్తున్నాను. నాకు తెలుగు భాష చాలా ఇష్టం.",
    "08_telugu_english": "నేను Red Hatలో Junior Consultantగా పనిచేస్తున్నాను. నా రోజువారీ పనిలో Linux, Kubernetes, OpenShift మరియు Cloud technologies ఉపయోగిస్తాను.",
}


def cases_as_json() -> list[dict[str, str]]:
    return [asdict(case) for case in CASES]
