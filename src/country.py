"""Country and work-mode normalization utilities."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

REMOTE_PATTERN = re.compile(r"(?<!non[- ])(?<!no )\bremote\b")
GLOBAL_PATTERN = re.compile(
    r"\b(anywhere|worldwide|global|everywhere|international|overseas)\b"
)
HYBRID_PATTERN = re.compile(r"\bhybrid\b")
US_AMBIGUOUS_CA = re.compile(r"\bca\b")

NOISE_EXACT = {
    "",
    "country",
    "unknown",
    "unemployed",
    "on-site",
    "onsite",
    "in-person",
    "in person",
    "flexible",
    "busser",
    "retail",
    "consulting",
    "construction",
    "masters",
    "recent graduate",
    "any",
    "any location",
    "seeking feedback",
    "resume review",
    "no interviews",
    "hurting morale",
    "unable to find a job",
    "please help with a resume template",
    "layout check and resume content analysis",
    "why am i not hearing back?",
    "4k+ applications in what am i doing wrong?",
    "100+ applications with no response",
}

REGION_PATTERNS: dict[str, list[str]] = {
    "Europe / EU": [
        r"\beurope(an union)?\b",
        r"\beu\b",
        r"\bue\b",
        r"\bemea\b",
        r"\bdach\b",
    ],
    "North America": [r"\bnorth america\b"],
    "Middle East / MENA": [r"\bmiddle\s*east\b", r"\bmena\b", r"\bgcc\b"],
    "Asia Pacific / APAC": [
        r"\bapac\b",
        r"\basia pacific\b",
        r"\bsouth asia\b",
        r"\basia\b",
    ],
    "Latin America / LATAM": [r"\blatam\b", r"\blatin america\b"],
    "Africa": [r"\bafrica\b"],
}

COUNTRY_PATTERNS: dict[str, list[str]] = {
    "Canada": [
        r"\bcanada\b",
        r"\bcan\b",
        r"\btoronto\b",
        r"\bgta\b",
        r"\bontario\b",
        r"\bon\s+ca\b",
        r"\bvancouver\b",
        r"\bbc\b",
        r"\bbritish columbia\b",
        r"\bmontreal\b",
        r"\bquebec\b",
        r"\bqc\b",
        r"\bottawa\b",
        r"\balberta\b",
        r"\bcalgary\b",
        r"\bedmonton\b",
        r"\bwinnipeg\b",
        r"\bsaskatoon\b",
        r"\bsaskatchewan\b",
        r"\bnewfoundland\b",
        r"\bst\.?\s*john['’]?s\b",
        r"\bscarborough\b",
        r"\bmississauga\b",
    ],
    "United States": [
        r"\busa\b",
        r"\bu\.s\.a\.?\b",
        r"\bu\.s\.?\b",
        r"\bi\.s\.?\b",
        r"\bus\b",
        r"\bunited\s+states(\s+of\s+america)?\b",
        r"\bunited\s+state\b",
        r"\bunited\s+stated\b",
        r"\bunites\s+states\b",
        r"\bamerica\b",
        r"\b(nyc|new york|nj|new jersey|california|cali|bay area|sf|los angeles|la|socal|sacramento|san jose|san diego|orange county)\b",
        r"\b(texas|tx|dfw|austin|houston|dallas|san antonio|florida|fl|miami|orlando|tampa|aventura|jacksonville|winter haven)\b",
        r"\b(illinois|chicago|chicagoland|atlanta|atl|virginia|va|maryland|md|dmv|dc|washington d\.?c\.?)\b",
        r"\b(seattle|pnw|oregon|portland|boston|massachusetts|massachussetts|ma|pennsylvania|pa|philadelphia|pittsburgh)\b",
        r"\b(north carolina|nc|charlotte|raleigh|chapel hill|south carolina|sc|michigan|detroit|ohio|cleveland|cincinnati|columbus)\b",
        r"\b(arizona|az|phoenix|tucson|colorado|denver|colorado springs|utah|salt lake city|minnesota|minneapolis|twin cities)\b",
        r"\b(indiana|indianapolis|terra haute|wisconsin|milwaukee|missouri|st\.?\s*louis|kansas|wichita|iowa|ia|ames)\b",
        r"\b(nevada|vegas|las vegas|tennessee|alabama|huntsville|louisiana|new orleans|connecticut|ct|vermont|new hampshire|hawaii|oklahoma|idaho|new mexico|arkansas|delaware|kentucky|west virginia|beaverton)\b",
    ],
    "India": [
        r"\bindia\b",
        r"\bind\b",
        r"\bbengaluru\b",
        r"\bbangalore\b",
        r"\bbanglore\b",
        r"\bbenguluru\b",
        r"\bmumbai\b",
        r"\b(new\s+)?delhi\b",
        r"\bncr\b",
        r"\bgurugram\b",
        r"\bnoida\b",
        r"\bpune\b",
        r"\bhyderabad\b",
        r"\bchennai\b",
        r"\bkolkata\b",
        r"\bahmedabad\b",
        r"\bindore\b",
        r"\bcoimbatore\b",
        r"\bmysuru\b",
        r"\bbhubaneswar\b",
        r"\bandhra\b",
    ],
    "United Kingdom": [
        r"\buk\b",
        r"\bu\.k\.?\b",
        r"\bunited\s+kingdo[mn]\b",
        r"\bbritain\b",
        r"\bgb\b",
        r"\bengland\b",
        r"\bscotland\b",
        r"\bwales\b",
        r"\bt?london\b",
        r"\bmanchester\b",
        r"\bsheffield\b",
        r"\bblackpool\b",
    ],
    "Australia": [
        r"\baustralia\b",
        r"\baus\b",
        r"\bsydney\b",
        r"\bmelbourne\b",
        r"\bmelb\b",
        r"\bbrisbane\b",
        r"\bperth\b",
        r"\bnsw\b",
    ],
    "Germany": [r"\bgermany\b", r"\bberlin\b", r"\bmunich\b", r"\bcologne\b"],
    "Philippines": [
        r"\bphilipp?ines?\b",
        r"\bphill?ipines?\b",
        r"\bph\b",
        r"\bmanila\b",
        r"\bcebu\b",
    ],
    "United Arab Emirates": [
        r"\buae\b",
        r"\bunited arab emirates\b",
        r"\bdubai\b",
        r"\babu dhabi\b",
    ],
    "Singapore": [r"\bsingapore\b", r"\bsg\b"],
    "Ireland": [r"\bireland\b", r"\bdublin\b", r"\bshannon\b", r"\bie\b"],
    "Netherlands": [r"\bnetherlands\b", r"\bholland\b", r"\bnl\b"],
    "France": [r"\bfrance\b", r"\bparis\b"],
    "Spain": [
        r"\bspain\b",
        r"\bmadrid\b",
        r"\bbarcelona\b",
        r"\bm[aá]laga\b",
        r"\bvalencia\b",
    ],
    "Italy": [r"\bitaly\b", r"\brome\b", r"\bmilan\b", r"\bcosenza\b"],
    "Poland": [r"\bpoland\b", r"\bwarsaw\b", r"\bpozna[nń]\b"],
    "Pakistan": [
        r"\bpakistant?\b",
        r"\bpk\b",
        r"\blahore\b",
        r"\bkarachi\b",
        r"\bislamabad\b",
        r"\brawalpindi\b",
    ],
    "Egypt": [r"\begypt\b", r"\begy\b", r"\beg\b"],
    "Nigeria": [r"\bnigeria\b"],
    "Brazil": [
        r"\bbrazil\b",
        r"\bbrasil\b",
        r"\brio de janeiro\b",
        r"\bbras[ií]lia\b",
        r"\bbr\b",
    ],
    "Mexico": [r"\bm[eé]xico\b", r"\btijuana\b"],
    "South Africa": [r"\bsouth africa\b", r"\bcape town\b"],
    "New Zealand": [r"\bnew zealand\b", r"\bnz\b"],
    "Nepal": [r"\bnepal\b", r"\bkathmandu\b"],
    "Turkey": [r"\bturkey\b", r"\bt[uü]rkiye\b", r"\bistanbul\b"],
    "Israel": [r"\bisrael\b"],
    "Saudi Arabia": [r"\bsaudi arabia\b", r"\briyadh\b", r"\bjeddah\b"],
    "Switzerland": [
        r"\bswitzerland\b",
        r"\bz[uü]rich\b",
        r"\bgeneva\b",
        r"\bch\b",
    ],
    "China": [r"\bchina\b"],
    "Hong Kong": [r"\bhong kong\b", r"\bhk\b"],
    "Romania": [r"\bromania\b", r"\bbucharest\b"],
    "Malaysia": [r"\bmalaysia\b"],
    "Indonesia": [r"\bindonesia\b"],
    "Thailand": [r"\bthailand\b"],
    "Japan": [r"\bjapan\b", r"\btokyo\b"],
    "South Korea": [r"\bsouth korea\b", r"\bkorea\b", r"\bseoul\b"],
    "Morocco": [r"\bmorocco\b", r"\bcasablanca\b", r"\brabat\b"],
    "Tunisia": [r"\btunisia\b"],
    "Algeria": [r"\balgeria\b"],
    "Lebanon": [r"\blebanon\b"],
    "Serbia": [r"\bserbia\b"],
    "Croatia": [r"\bcroatia\b"],
    "Bosnia and Herzegovina": [r"\bbosnia and herzegovina\b", r"\bbih\b"],
    "North Macedonia": [r"\b(north )?macedonia\b"],
    "Hungary": [r"\bhungary\b", r"\bbudapest\b"],
    "Czechia": [r"\bczechia\b", r"\bczech republic\b", r"\bprague\b"],
    "Austria": [r"\baustria\b"],
    "Belgium": [r"\bbelgium\b", r"\bbrussels\b", r"\bbe\b"],
    "Denmark": [r"\bdenmark\b"],
    "Sweden": [r"\bsweden\b", r"\bgothenburg\b", r"\bse\b"],
    "Norway": [r"\bnorway\b"],
    "Finland": [r"\bfinland\b"],
    "Greece": [r"\bgreece\b", r"\bathens\b", r"\bmykonos\b"],
    "Portugal": [r"\bportugal\b", r"\blisbon\b"],
    "Luxembourg": [r"\bluxembourg\b"],
    "Ukraine": [r"\bukraine\b"],
    "Russia": [r"\brussia\b"],
    "Bangladesh": [r"\bbangladesh\b", r"\bbd\b"],
    "Sri Lanka": [r"\bsri lanka\b"],
    "Vietnam": [r"\bvietnam\b"],
    "Taiwan": [r"\btaiwan\b"],
    "Qatar": [r"\bqatar\b"],
    "Kuwait": [r"\bkuwait\b"],
    "Oman": [r"\boman\b"],
    "Jordan": [r"\bjordan\b"],
    "Iraq": [r"\biraq\b"],
    "Syria": [r"\bsyria\b"],
    "Kenya": [r"\bkenya\b"],
    "Ethiopia": [r"\bethiopia\b", r"\baddis ababa\b"],
    "Argentina": [r"\bargentina\b"],
    "Colombia": [r"\bcolombia\b"],
    "Venezuela": [r"\bvenezuela\b"],
    "Costa Rica": [r"\bcosta rica\b"],
    "Peru": [r"\bperu\b"],
    "Jamaica": [r"\bjamaica\b"],
    "Trinidad and Tobago": [
        r"\btrinidad( &| and)? tobago\b",
        r"\btrinidad\b",
    ],
    "Dominican Republic": [r"\bdominican republic\b"],
    "Armenia": [r"\byerevan\b", r"\barmenia\b"],
}

_COMPILED_REGION_PATTERNS = {
    region: [re.compile(p) for p in patterns]
    for region, patterns in REGION_PATTERNS.items()
}
_COMPILED_COUNTRY_PATTERNS = {
    country: [re.compile(p) for p in patterns]
    for country, patterns in COUNTRY_PATTERNS.items()
}


def standardize_location(val: object) -> Optional[str]:
    """Normalize a raw country/location string into a standardized label."""
    if pd.isna(val):
        return None
    s = str(val).strip().strip('"').strip().lower()
    if s in NOISE_EXACT:
        return None
    if REMOTE_PATTERN.search(s):
        return "Remote"
    if GLOBAL_PATTERN.search(s):
        return "Global / Anywhere"
    if HYBRID_PATTERN.search(s):
        return "Hybrid"
    if "->" in s or ">" in s:
        s = re.split(r"-?>", s)[-1].strip()
        if GLOBAL_PATTERN.search(s):
            return "Global / Anywhere"
    for region, patterns in _COMPILED_REGION_PATTERNS.items():
        if any(p.search(s) for p in patterns):
            return region
    matched = [
        country
        for country, patterns in _COMPILED_COUNTRY_PATTERNS.items()
        if any(p.search(s) for p in patterns)
    ]
    if (
        "Canada" not in matched
        and "United States" not in matched
        and US_AMBIGUOUS_CA.search(s)
    ):
        matched.append("United States")
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return " / ".join(matched)
    return "UNMAPPED"


def build_country_mapping(series: pd.Series) -> dict[str, Optional[str]]:
    """Build a mapping from unique raw country strings to normalized codes."""
    return {v: standardize_location(v) for v in series.dropna().unique()}
