"""Synthetic PII generation and trace injection for privacy testing."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, fields

from agentlens.schema.trace import SessionTrace


@dataclass
class SyntheticPII:
    """A bundle of synthetic personally identifiable information."""

    full_name: str = ""
    email: str = ""
    phone: str = ""
    ssn: str = ""
    api_key: str = ""
    credit_card: str = ""
    ip_address: str = ""
    github_username: str = ""
    home_address: str = ""
    code_snippet: str = ""
    file_path: str = ""
    aws_secret: str = ""


# Name pools by region for diversity
_FIRST_NAMES = {
    "western": ["James", "Emily", "Michael", "Sarah", "Robert", "Jessica"],
    "south_asian": ["Priya", "Rahul", "Ananya", "Vikram", "Deepa", "Arjun"],
    "east_asian": ["Wei", "Yuki", "Hiroshi", "Mei", "Takeshi", "Sakura"],
    "arabic": ["Fatima", "Omar", "Aisha", "Khalid", "Layla", "Hassan"],
    "african": ["Amara", "Kwame", "Zara", "Tendai", "Nia", "Kofi"],
    "latin_american": ["Sofia", "Mateo", "Valentina", "Santiago", "Camila", "Diego"],
}

_LAST_NAMES = {
    "western": ["Johnson", "Williams", "Brown", "Taylor", "Anderson", "Martinez"],
    "south_asian": ["Sharma", "Patel", "Gupta", "Singh", "Kumar", "Reddy"],
    "east_asian": ["Chen", "Tanaka", "Yamamoto", "Wang", "Suzuki", "Park"],
    "arabic": ["Al-Rashid", "Mansour", "Nasser", "Haddad", "Khoury", "Farouk"],
    "african": ["Okafor", "Mensah", "Diallo", "Mwangi", "Adeyemi", "Banda"],
    "latin_american": ["Rodriguez", "Gonzalez", "Hernandez", "Lopez", "Morales", "Reyes"],
}

_STREETS = [
    "Oak Street", "Maple Avenue", "Cedar Lane", "Pine Road",
    "Elm Boulevard", "Birch Drive", "Walnut Court", "Spruce Way",
]

_CITIES = [
    "Springfield, IL 62704", "Portland, OR 97201", "Austin, TX 78701",
    "Denver, CO 80202", "Seattle, WA 98101", "Boston, MA 02101",
]

_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "fastmail.com"]


class PIIGenerator:
    """Generates diverse synthetic PII bundles for privacy testing."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._generated_emails: set[str] = set()

    def generate(self, count: int = 50) -> list[SyntheticPII]:
        """Generate `count` unique synthetic PII bundles."""
        results: list[SyntheticPII] = []
        regions = list(_FIRST_NAMES.keys())

        for i in range(count):
            region = regions[i % len(regions)]
            pii = self._generate_one(region, i)
            results.append(pii)

        return results

    def _generate_one(self, region: str, index: int) -> SyntheticPII:
        rng = self._rng

        first = rng.choice(_FIRST_NAMES[region])
        last = rng.choice(_LAST_NAMES[region])
        full_name = f"{first} {last}"

        # Ensure unique email
        local_part = f"{first.lower()}.{last.lower()}{rng.randint(10, 99)}"
        domain = rng.choice(_DOMAINS)
        email = f"{local_part}@{domain}"
        while email in self._generated_emails:
            local_part = f"{first.lower()}.{last.lower()}{rng.randint(100, 999)}"
            email = f"{local_part}@{domain}"
        self._generated_emails.add(email)

        phone = f"+1-{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
        ssn = f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"

        api_key_chars = string.ascii_letters + string.digits
        api_key = "sk-" + "".join(rng.choices(api_key_chars, k=40))

        cc_groups = [str(rng.randint(1000, 9999)) for _ in range(4)]
        credit_card = "-".join(cc_groups)

        ip_parts = [
            rng.randint(10, 223), rng.randint(0, 255),
            rng.randint(0, 255), rng.randint(1, 254),
        ]
        ip_address = ".".join(str(p) for p in ip_parts)

        github_username = f"{first.lower()}{last.lower()}{rng.randint(1, 999)}"

        street_num = rng.randint(100, 9999)
        street = rng.choice(_STREETS)
        city = rng.choice(_CITIES)
        home_address = f"{street_num} {street}, {city}"

        var_name = rng.choice(["user_password", "db_connection", "secret_token", "api_secret"])
        var_val = "".join(rng.choices(string.ascii_letters + string.digits, k=16))
        code_snippet = f'{var_name} = "{var_val}"'

        dirs = rng.choice(["/home", "/Users", "C:\\Users"])
        username = first.lower()
        proj_name = "".join(rng.choices(string.ascii_lowercase, k=8))
        file_path = f"{dirs}/{username}/projects/{proj_name}/config.env"

        aws_secret = "".join(rng.choices(string.ascii_letters + string.digits + "+/", k=40))

        return SyntheticPII(
            full_name=full_name,
            email=email,
            phone=phone,
            ssn=ssn,
            api_key=api_key,
            credit_card=credit_card,
            ip_address=ip_address,
            github_username=github_username,
            home_address=home_address,
            code_snippet=code_snippet,
            file_path=file_path,
            aws_secret=aws_secret,
        )

    def inject_into_trace(self, trace: SessionTrace, pii: SyntheticPII) -> SessionTrace:
        """Return a modified copy of the trace with PII injected. Original is unchanged."""
        data = trace.model_dump()

        # Inject into action output_summary (prepend PII to survive 500-char truncation)
        for action in data["actions"]:
            pii_prefix = (
                f"Contact {pii.full_name} at {pii.email}, "
                f"key={pii.api_key[:20]}, {pii.code_snippet}"
            )
            original = action["output_summary"]
            combined = f"{pii_prefix} | {original}"
            # Respect 500-char field limit (validator truncates at 497+...)
            action["output_summary"] = combined[:500]

        # Inject into escalation descriptions (short PII, fit 200-char limit)
        for esc in data.get("escalations", []):
            short_pii = f"User {pii.email} ({pii.phone})"
            original = esc["description"]
            combined = f"{short_pii} - {original}"
            esc["description"] = combined[:200]

        # Inject into metadata
        data["metadata"]["user_context"] = pii.home_address

        return SessionTrace(**data)

    @staticmethod
    def get_all_pii_strings(pii: SyntheticPII) -> set[str]:
        """Return a flat set of PII strings including partial matches."""
        strings: set[str] = set()

        for f in fields(pii):
            val = getattr(pii, f.name)
            if val:
                strings.add(val)

        # Partial matches
        name_parts = pii.full_name.split()
        for part in name_parts:
            if len(part) >= 3:
                strings.add(part)

        # Email local part
        if "@" in pii.email:
            strings.add(pii.email.split("@")[0])

        # First 8 chars of API key (after sk- prefix)
        if len(pii.api_key) > 11:
            strings.add(pii.api_key[:11])  # "sk-" + 8 chars

        return strings
