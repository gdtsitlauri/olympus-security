"""OLYMPUS — Offensive and Defensive Autonomous Security Intelligence System."""

__version__ = "1.0.0"
__author__ = "OLYMPUS Team"
__license__ = "MIT"

from olympus.core.orchestrator import ORCHESTRATOR
from olympus.core.knowledge_base import KB

# Register all modules on import
def _register_all() -> None:
    from olympus.modules.module1_pentest import PentestModule
    from olympus.modules.module2_virus import VirusDetectionModule
    from olympus.modules.module3_zeroday import ZeroDayModule
    from olympus.modules.module4_threat_intel import ThreatIntelligenceModule
    from olympus.modules.module5_deception import DeceptionModule
    from olympus.modules.module6_evolution import SelfEvolutionModule
    from olympus.modules.module7_social_eng import SocialEngDetectionModule
    from olympus.modules.module8_ai_integrity import AIIntegrityModule
    from olympus.modules.module9_llm_defense import LLMDefenseModule
    from olympus.modules.module10_forensics import ForensicsModule

    for Module in [
        PentestModule, VirusDetectionModule, ZeroDayModule,
        ThreatIntelligenceModule, DeceptionModule, SelfEvolutionModule,
        SocialEngDetectionModule, AIIntegrityModule, LLMDefenseModule,
        ForensicsModule,
    ]:
        ORCHESTRATOR.register(Module())


_register_all()

__all__ = ["ORCHESTRATOR", "KB", "__version__"]
