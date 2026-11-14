"""Fixed Telugu, Tenglish and English prompts for repeatable API benchmarks.

The corpus is deliberately small and stable so two builds can be compared on the
same text.  It is not a quality certificate: pronunciation, prosody and speaker
similarity still require human listening.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


TELUGU = "Telugu"
MIXED = "Telugu + English"
ENGLISH = "English"


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    text: str
    language: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


TELUGU_CASES = (
    Case("te001", "conversation", "ఈ రోజు ఉదయం వాతావరణం చాలా ఆహ్లాదంగా ఉంది.", TELUGU),
    Case("te002", "conversation", "మీరు భోజనం చేశారా? నేను ఇప్పుడే చేశాను.", TELUGU),
    Case("te003", "conversation", "సాయంత్రం మనం కలిసి నడకకు వెళ్దామా?", TELUGU),
    Case("te004", "conversation", "నాకు ఒక కప్పు వేడి కాఫీ కావాలి.", TELUGU),
    Case("te005", "conversation", "రేపు మీకు సమయం ఉంటే నాకు చెప్పండి.", TELUGU),
    Case("te006", "formal", "దయచేసి ఈ పత్రాన్ని జాగ్రత్తగా పరిశీలించండి.", TELUGU),
    Case("te007", "formal", "మీ సహకారానికి హృదయపూర్వక ధన్యవాదాలు.", TELUGU),
    Case("te008", "formal", "సమావేశం ప్రారంభమయ్యే ముందు అందరూ తమ స్థానాల్లో కూర్చోవాలి.", TELUGU),
    Case("te009", "vowels_aspiration", "ఆకాశంలో ఎగురుతున్న పక్షులను పిల్లలు ఆసక్తిగా చూశారు.", TELUGU),
    Case("te010", "vowels_aspiration", "ఖాళీ పాత్రను శుభ్రంగా కడిగి అలమారలో పెట్టండి.", TELUGU),
    Case("te011", "vowels_aspiration", "భారీ వర్షం పడినా రైతులు పొలానికి వెళ్లారు.", TELUGU),
    Case("te012", "vowels_aspiration", "దీర్ఘకాలిక ఆరోగ్యానికి ప్రశాంతమైన నిద్ర అవసరం.", TELUGU),
    Case("te013", "conjuncts", "స్వాతంత్ర్యం ప్రతి పౌరుడి బాధ్యతను గుర్తు చేస్తుంది.", TELUGU),
    Case("te014", "conjuncts", "జ్ఞానం మరియు విజ్ఞానం సమాజ అభివృద్ధికి తోడ్పడతాయి.", TELUGU),
    Case("te015", "conjuncts", "కృష్ణుడు క్షమతో సమస్యను పరిష్కరించాడు.", TELUGU),
    Case("te016", "conjuncts", "పుస్తకంలోని ప్రశ్నలకు విద్యార్థులు సమాధానాలు రాశారు.", TELUGU),
    Case("te017", "similar_sounding", "పాలు తీసుకుని బాలుడికి ఇచ్చి, పక్కనే ఉన్న పలకను శుభ్రం చేశాను.", TELUGU),
    Case("te018", "similar_sounding", "కళాకారుడు కలతో కొత్త చిత్రాన్ని గీశాడు.", TELUGU),
    Case("te019", "similar_sounding", "చిరు జల్లులో చిన్నారి చిరునవ్వుతో ఆడుకుంది.", TELUGU),
    Case("te020", "similar_sounding", "నది ఒడ్డున నిలబడి నీటి అలలను చూశాము.", TELUGU),
    Case("te021", "punctuation", "అమ్మా, మీరు ఎప్పుడు వస్తారు? మేము ఎదురుచూస్తున్నాము.", TELUGU),
    Case("te022", "punctuation", "అద్భుతం! ఆ చిన్నారి ఎంత చక్కగా పాడింది.", TELUGU),
    Case("te023", "punctuation", "అతను అన్నాడు: “నిజాయితీ ఎప్పుడూ గెలుస్తుంది.”", TELUGU),
    Case("te024", "numbers", "రేపు ఉదయం 10 గంటలకు సమావేశం ప్రారంభమవుతుంది.", TELUGU),
    Case("te025", "numbers", "ఈ పుస్తకానికి 250 రూపాయలు చెల్లించాను.", TELUGU),
    Case("te026", "numbers", "మా గ్రామం నగరానికి దాదాపు 18 కిలోమీటర్ల దూరంలో ఉంది.", TELUGU),
    Case("te027", "names", "కౌశిక్ విజయవాడ నుంచి ఢిల్లీకి ప్రయాణించాడు.", TELUGU),
    Case("te028", "names", "సీత, రమేష్ మరియు అనన్య గ్రంథాలయంలో కలుసుకున్నారు.", TELUGU),
    Case("te029", "long_sentence", "మనం ఏదైనా పని ప్రారంభించే ముందు దాని గురించి పూర్తిగా అర్థం చేసుకుని, సరైన విధంగా ప్రణాళిక చేసుకుంటే ఆ పనిని మరింత సమర్థవంతంగా పూర్తి చేయగలం.", TELUGU),
    Case("te030", "long_sentence", "ఉదయం త్వరగా లేచి వ్యాయామం చేసిన తరువాత అల్పాహారం తీసుకుని, రోజంతా చేయాల్సిన పనులను క్రమంగా పూర్తి చేయాలని ఆమె నిర్ణయించుకుంది.", TELUGU),
)

MIXED_CASES = (
    Case("mix001", "mixed_technical", "నేను Linux కంప్యూటర్‌లో కొత్త Python ప్రోగ్రామ్‌ను పరీక్షిస్తున్నాను.", MIXED),
    Case("mix002", "mixed_technical", "Red Hat బృందం Kubernetes క్లస్టర్‌ను జాగ్రత్తగా నిర్వహిస్తోంది.", MIXED),
    Case("mix003", "mixed_technical", "ఈ API ద్వారా Cloud సేవకు సురక్షితంగా కనెక్ట్ కావచ్చు.", MIXED),
    Case("mix004", "mixed_conversation", "ఈ రోజు online meeting ఉందా, లేక రేపటికి మార్చారా?", MIXED),
    Case("mix005", "mixed_conversation", "నేను message పంపాను, మీరు వచ్చినప్పుడు reply ఇవ్వండి.", MIXED),
    Case("mix006", "mixed_names", "Ananya హైదరాబాద్‌లోని office నుంచి Zoom callలో చేరింది.", MIXED),
    Case("mix007", "mixed_technical", "కొత్త software update తర్వాత database backup సరిగ్గా ఉందో చూడండి.", MIXED),
    Case("mix008", "mixed_long_sentence", "ప్రాజెక్ట్ గడువులోపు పూర్తిచేయడానికి team రోజూ progressను పంచుకుని, అవసరమైనప్పుడు designలో చిన్న మార్పులు చేస్తుంది.", MIXED),
)

# Optional Chatterbox-Telugu checks.  These are marked experimental by the API
# and must not be reported as validated English quality without listening tests.
ENGLISH_CASES = (
    Case("en001", "english_conversation", "Hello, how can I help you today?", ENGLISH),
    Case("en002", "english_numbers", "The meeting starts at 10:30 in room 204.", ENGLISH),
    Case("en003", "english_technical", "The Kubernetes cluster is healthy and all pods are running.", ENGLISH),
    Case("en004", "english_names", "Ananya and Kaushik will join the call from Hyderabad.", ENGLISH),
    Case("en005", "english_punctuation", "Really? That is excellent news — thank you!", ENGLISH),
    Case("en006", "english_long", "Before starting any task, take a moment to understand the requirements, plan the work, and verify the result carefully.", ENGLISH),
)

CASES = TELUGU_CASES + MIXED_CASES + ENGLISH_CASES
CASES_BY_ID = {case.case_id: case for case in CASES}


def select_cases(case_ids: list[str] | None = None, limit: int | None = None) -> list[Case]:
    selected = list(CASES)
    if case_ids:
        missing = [case_id for case_id in case_ids if case_id not in CASES_BY_ID]
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(missing)}")
        selected = [CASES_BY_ID[case_id] for case_id in case_ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected
