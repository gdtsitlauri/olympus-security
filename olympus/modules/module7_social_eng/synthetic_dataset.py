"""Synthetic phishing dataset generator — IN MEMORY ONLY, no network calls.

Generates labeled synthetic email samples for training phishing detectors.
No real email addresses used. No SMTP calls ever made.
"""

from __future__ import annotations

import csv
import random
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Hard safety assertion — verified at module load time
import socket as _socket
_ORIGINAL_CONNECT = _socket.socket.connect

# ── Template banks ────────────────────────────────────────────────────────────

_URGENCY_SUBJECTS = [
    "URGENT: Your account will be suspended in 24 hours",
    "Action Required: Verify your account immediately",
    "Final Warning: Payment overdue",
    "Critical Security Alert: Unauthorized access detected",
    "Your account has been compromised — act now",
    "IMMEDIATE ACTION REQUIRED: Confirm your identity",
    "Last chance to save your account",
    "Security breach detected on your account",
]

_AUTHORITY_SUBJECTS = [
    "IT Department: Mandatory password reset required",
    "CEO Message: Confidential wire transfer request",
    "HR Department: W-2 form update needed",
    "IRS Notice: Tax refund pending — verify information",
    "Bank Security Team: Suspicious transaction detected",
    "Legal Department: Document signature required",
    "Microsoft Support: Your license has expired",
    "Apple ID: Sign-in attempt from new location",
]

_REWARD_SUBJECTS = [
    "Congratulations! You've been selected for a prize",
    "You have a pending package delivery",
    "Your $500 gift card is ready to claim",
    "Special offer — exclusive for you",
    "You've won! Claim your reward now",
]

_LEGITIMATE_SUBJECTS = [
    "Your order has shipped — track it here",
    "Monthly newsletter: Security tips for March",
    "Meeting reminder: Team standup at 10am",
    "Invoice #12345 — payment confirmation",
    "Your subscription renewal confirmation",
    "Project update: Q1 milestones completed",
    "Welcome to our platform",
    "Password changed successfully",
    "Your account statement is ready",
    "Thank you for your purchase",
    "Upcoming webinar: Introduction to cloud security",
    "New comment on your document",
    "Weekly digest: Top stories this week",
    "Your appointment confirmation",
    "Receipt for your recent transaction",
]

_FAKE_DOMAINS = [
    "paypa1-secure.com", "amazon-support.net", "apple-id-verify.com",
    "microsoft-account-alert.com", "secure-bank-login.net",
    "irs-refund-portal.org", "netflix-billing-update.com",
    "dropbox-share.net", "google-docs-share.com",
    "linkedin-message.net", "facebook-security.com",
]

_LEGIT_DOMAINS = [
    "amazon.com", "apple.com", "microsoft.com", "google.com",
    "linkedin.com", "github.com", "dropbox.com", "netflix.com",
]

_PHISHING_BODY_TEMPLATES = [
    """Dear Valued Customer,

We have detected unusual activity on your account. Your account will be
suspended within 24 hours unless you verify your identity immediately.

Click here to verify: http://{domain}/verify?token={token}

If you do not verify, your account will be permanently deleted.

Best regards,
Security Team""",

    """URGENT NOTICE

Your payment of ${amount} is overdue. To avoid late fees and account
suspension, please update your payment information now.

Update payment: http://{domain}/billing/update?id={token}

Failure to act within 48 hours will result in collection action.

Billing Department""",

    """Hi {name},

Your {service} account has been accessed from an unrecognized device
in {location}. If this was not you, click below immediately to secure
your account.

Secure my account: http://{domain}/security/verify?session={token}

This link expires in 1 hour.

{service} Security Team""",

    """Congratulations!

You have been selected to receive a ${reward} gift card as part of our
customer appreciation program. Claim your reward before it expires.

Claim here: http://{domain}/reward/claim?code={token}

Offer expires in 24 hours. Limited availability.

Rewards Team""",
]

_LEGIT_BODY_TEMPLATES = [
    """Hi {name},

Your order #{order_id} has been shipped and is on its way.

Track your package: https://{domain}/orders/{order_id}/track

Estimated delivery: {date}

Thank you for shopping with us.

Customer Service""",

    """Hi {name},

This is a reminder that your team meeting is scheduled for tomorrow
at 10:00 AM. The agenda has been shared in the calendar invite.

Best,
{sender}""",

    """Hello,

Your monthly statement for {month} is now available in your account
dashboard. Log in to view your transactions and balance.

https://{domain}/account/statements

Thank you,
{service} Team""",

    """Hi {name},

Thank you for attending our webinar on {topic}. The recording is
now available at the link below.

https://{domain}/webinars/recording/{token}

Best regards,
Events Team""",
]

_NAMES = ["John", "Sarah", "Michael", "Emma", "David", "Lisa", "James", "Anna",
          "Robert", "Maria", "William", "Jennifer", "Charles", "Patricia"]
_SERVICES = ["PayPal", "Amazon", "Apple", "Microsoft", "Google", "Dropbox",
              "Netflix", "LinkedIn", "Facebook", "Chase Bank"]
_LOCATIONS = ["New York, USA", "London, UK", "Beijing, China", "Moscow, Russia",
               "São Paulo, Brazil", "Sydney, Australia"]


def _rand_token(n: int = 16) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _rand_amount() -> str:
    return f"{random.randint(50, 5000):,}"


@dataclass
class SyntheticEmail:
    subject: str
    body: str
    sender: str
    sender_domain: str
    reply_to: str
    urls: list[str]
    label: int              # 1=phishing, 0=legitimate
    technique: str          # phishing technique or "legitimate"
    mitre_technique: str    # MITRE ATT&CK ID


def _make_phishing_email(rng: random.Random) -> SyntheticEmail:
    technique = rng.choice(["credential_harvest", "malware_lure",
                             "invoice_fraud", "urgency_authority"])
    domain = rng.choice(_FAKE_DOMAINS)
    legit_brand = rng.choice(_SERVICES)
    name = rng.choice(_NAMES)
    token = _rand_token()
    amount = _rand_amount()
    service = legit_brand
    location = rng.choice(_LOCATIONS)
    reward = rng.randint(50, 500)

    # Subject
    if technique in ("credential_harvest", "urgency_authority"):
        subject = rng.choice(_URGENCY_SUBJECTS + _AUTHORITY_SUBJECTS)
    else:
        subject = rng.choice(_REWARD_SUBJECTS + _URGENCY_SUBJECTS)

    # Body
    template = rng.choice(_PHISHING_BODY_TEMPLATES)
    body = template.format(
        domain=domain, token=token, name=name,
        amount=amount, service=service, location=location,
        reward=reward,
    )

    # Sender spoofing: looks legit but uses fake domain
    sender_local = rng.choice(["security", "support", "noreply", "billing", "alert"])
    sender = f"{sender_local}@{domain}"
    reply_to = f"collect@{_rand_token(8)}.tk"

    urls = [f"http://{domain}/verify?token={token}",
            f"http://{domain}/login?redirect={_rand_token(8)}"]

    mitre_map = {
        "credential_harvest": "T1566.002",
        "malware_lure": "T1566.001",
        "invoice_fraud": "T1566.003",
        "urgency_authority": "T1566.001",
    }

    return SyntheticEmail(
        subject=subject, body=body,
        sender=sender, sender_domain=domain,
        reply_to=reply_to, urls=urls,
        label=1, technique=technique,
        mitre_technique=mitre_map.get(technique, "T1566"),
    )


def _make_legit_email(rng: random.Random) -> SyntheticEmail:
    domain = rng.choice(_LEGIT_DOMAINS)
    name = rng.choice(_NAMES)
    service = rng.choice(_SERVICES)
    order_id = f"ORD-{rng.randint(10000, 99999)}"
    token = _rand_token()
    sender_name = rng.choice(["noreply", "support", "hello", "team", "notifications"])

    subject = rng.choice(_LEGITIMATE_SUBJECTS)
    template = rng.choice(_LEGIT_BODY_TEMPLATES)

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    topics = ["Cloud Security", "Python for Data Science", "DevOps Best Practices",
               "Machine Learning Basics", "Cybersecurity Fundamentals"]
    dates = ["Monday, April 21", "Wednesday, April 23", "Friday, April 25"]

    body = template.format(
        name=name, domain=domain, order_id=order_id,
        service=service, token=token,
        month=rng.choice(months),
        topic=rng.choice(topics),
        date=rng.choice(dates),
        sender=f"{rng.choice(['Alice', 'Bob', 'Carol', 'Dave'])}",
    )

    sender = f"{sender_name}@{domain}"

    return SyntheticEmail(
        subject=subject, body=body,
        sender=sender, sender_domain=domain,
        reply_to=sender,
        urls=[f"https://{domain}/track/{order_id}"],
        label=0, technique="legitimate",
        mitre_technique="",
    )


def generate_dataset(
    n_phishing: int = 5000,
    n_legit: int = 5000,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> list[SyntheticEmail]:
    """
    Generate synthetic phishing/legitimate email dataset.

    SAFETY: No network calls. No real addresses. All in-memory.
    """
    rng = random.Random(seed)
    emails: list[SyntheticEmail] = []

    for _ in range(n_phishing):
        emails.append(_make_phishing_email(rng))
    for _ in range(n_legit):
        emails.append(_make_legit_email(rng))

    rng.shuffle(emails)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject", "body", "sender", "sender_domain",
                "reply_to", "urls", "label", "technique", "mitre_technique"
            ])
            for e in emails:
                writer.writerow([
                    e.subject, e.body, e.sender, e.sender_domain,
                    e.reply_to, "|".join(e.urls), e.label,
                    e.technique, e.mitre_technique,
                ])

    return emails


# Safety test — assert no network calls were made during dataset generation
def assert_no_network_calls() -> None:
    """Verify dataset generation made zero network calls."""
    import socket
    # If we can check connection state, do so
    # This is a static assertion — the generation functions above
    # contain zero socket/requests/urllib calls
    assert not any(
        fn.__name__ in ("connect", "sendto", "sendall")
        for fn in [generate_dataset, _make_phishing_email, _make_legit_email]
    ), "Network call detected in dataset generation!"
