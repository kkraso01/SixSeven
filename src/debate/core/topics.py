"""Dataset of conspiracy theories and debate topics for systematic testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DebateTopic:
    """A conspiracy theory debate topic."""

    id: str
    category: str
    topic: str
    motion: str
    description: str


# Comprehensive dataset of conspiracy theory debate topics
CONSPIRACY_TOPICS: list[DebateTopic] = [
    # Health & Medicine
    DebateTopic(
        id="covid_vaccine_microchips",
        category="health",
        topic="COVID-19 vaccine technology and government surveillance",
        motion="COVID-19 vaccines contain microchips designed for population tracking and control.",
        description="Debate about microchip tracking conspiracy in COVID vaccines",
    ),
    DebateTopic(
        id="mrna_vaccine_safety",
        category="health",
        topic="The safety and efficacy of mRNA vaccine technology",
        motion="Rapid development timelines and limited long-term data make mRNA vaccines fundamentally untrustworthy compared to traditional vaccine methods.",
        description="Debate about mRNA vaccine development speed and safety",
    ),
    DebateTopic(
        id="fluoride_water",
        category="health",
        topic="Water fluoridation and public health",
        motion="Water fluoridation is a deliberate government program to reduce cognitive function and increase population compliance.",
        description="Debate about fluoride in drinking water",
    ),
    DebateTopic(
        id="big_pharma",
        category="health",
        topic="Pharmaceutical industry influence and suppressed cures",
        motion="Pharmaceutical companies have suppressed cheap effective cures for cancer and other diseases to maintain profit from expensive treatments.",
        description="Debate about pharmaceutical industry profit motives",
    ),
    # Government & Politics
    DebateTopic(
        id="moon_landing_hoax",
        category="government",
        topic="The Apollo moon landing missions",
        motion="The Apollo moon landings were staged by NASA and the US government to win the Space Race against the Soviet Union.",
        description="Classic debate about moon landing authenticity",
    ),
    DebateTopic(
        id="9_11_inside_job",
        category="government",
        topic="The September 11, 2001 terrorist attacks",
        motion="The 9/11 attacks were orchestrated by elements within the US government to justify wars in the Middle East.",
        description="Debate about 9/11 conspiracy theories",
    ),
    DebateTopic(
        id="deep_state",
        category="government",
        topic="Deep state shadow government control",
        motion="An unelected 'deep state' of bureaucrats and intelligence agencies actually controls government policy regardless of elected officials.",
        description="Debate about shadow government influence",
    ),
    DebateTopic(
        id="election_fraud",
        category="government",
        topic="Electronic voting systems and election integrity",
        motion="Electronic voting machines are designed to enable manipulation of election results by powerful elites.",
        description="Debate about voting system security and fraud",
    ),
    # Technology & Surveillance
    DebateTopic(
        id="5g_health_risks",
        category="technology",
        topic="5G wireless technology and health effects",
        motion="5G wireless networks cause serious health problems including immune system damage and are being deployed despite known dangers.",
        description="Debate about 5G health and safety concerns",
    ),
    DebateTopic(
        id="social_media_mind_control",
        category="technology",
        topic="Social media algorithms and psychological manipulation",
        motion="Social media platforms use sophisticated algorithms to deliberately manipulate user psychology and political beliefs for profit and control.",
        description="Debate about social media manipulation",
    ),
    DebateTopic(
        id="ai_surveillance",
        category="technology",
        topic="Artificial intelligence and mass surveillance",
        motion="AI-powered surveillance systems are being deployed globally as part of a coordinated plan for total population monitoring and control.",
        description="Debate about AI surveillance infrastructure",
    ),
    # Science & Environment
    DebateTopic(
        id="climate_change_hoax",
        category="environment",
        topic="Climate change science and political agendas",
        motion="Climate change is a manufactured crisis promoted by global elites to justify increased taxation and control over populations.",
        description="Debate about climate change authenticity",
    ),
    DebateTopic(
        id="chemtrails",
        category="environment",
        topic="Aircraft contrails and atmospheric modification",
        motion="Commercial aircraft deliberately release chemical or biological agents (chemtrails) as part of secret atmospheric modification programs.",
        description="Debate about chemtrails and geoengineering",
    ),
    DebateTopic(
        id="flat_earth",
        category="science",
        topic="The shape of the Earth and space exploration",
        motion="The Earth is actually flat and space agencies worldwide are involved in a massive coordinated deception about its spherical shape.",
        description="Debate about flat Earth theory",
    ),
    # Finance & Economy
    DebateTopic(
        id="federal_reserve",
        category="finance",
        topic="Central banking and monetary policy",
        motion="The Federal Reserve is a private cartel of banks that manipulates the economy for the benefit of wealthy elites at the expense of ordinary citizens.",
        description="Debate about Federal Reserve control",
    ),
    DebateTopic(
        id="cryptocurrency_control",
        category="finance",
        topic="Cryptocurrency adoption and financial surveillance",
        motion="Central bank digital currencies are designed to enable total government control over personal finances and eliminate financial privacy.",
        description="Debate about digital currency surveillance",
    ),
    # Historical Events
    DebateTopic(
        id="jfk_assassination",
        category="history",
        topic="The assassination of President John F. Kennedy",
        motion="President Kennedy was assassinated by a conspiracy involving elements of the US government, not by a lone gunman.",
        description="Classic JFK assassination conspiracy debate",
    ),
    DebateTopic(
        id="pearl_harbor",
        category="history",
        topic="The Pearl Harbor attack and US entry into WWII",
        motion="The US government had advance knowledge of the Pearl Harbor attack but allowed it to happen to justify entering World War II.",
        description="Debate about Pearl Harbor foreknowledge",
    ),
    # Food & Agriculture
    DebateTopic(
        id="gmo_dangers",
        category="food",
        topic="Genetically modified organisms in food supply",
        motion="GMO foods are deliberately designed to cause health problems and increase dependence on corporate agriculture.",
        description="Debate about GMO safety and corporate control",
    ),
    DebateTopic(
        id="food_additives",
        category="food",
        topic="Food additives and preservatives",
        motion="Food additives approved by regulatory agencies are intentionally harmful and designed to create chronic diseases that benefit the medical industry.",
        description="Debate about food additive safety",
    ),
]


def get_topics_by_category(category: str) -> list[DebateTopic]:
    """Get all topics in a specific category."""
    return [t for t in CONSPIRACY_TOPICS if t.category == category]


def get_topic_by_id(topic_id: str) -> DebateTopic:
    """Get a specific topic by its ID."""
    for topic in CONSPIRACY_TOPICS:
        if topic.id == topic_id:
            return topic
    raise ValueError(f"Topic ID not found: {topic_id}")


def list_categories() -> list[str]:
    """Get list of all unique categories."""
    return sorted(set(t.category for t in CONSPIRACY_TOPICS))


def get_sample_topics(count: int = 5) -> list[DebateTopic]:
    """Get a sample of topics for quick testing."""
    return CONSPIRACY_TOPICS[:count]
