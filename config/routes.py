"""
config/routes.py — Semantic Route Definitions.

This is where document classes are defined.
Each class is represented by a set of natural language descriptions.
The embedding model projects these descriptions into vector space
and computes a centroid per class.

+============================================================+
|  TO ADD A NEW CLASS:                                       |
|  1. Add a member to DocumentType                           |
|  2. Add a RoutePrototype with 4-8 descriptions             |
|  3. That's it. No other code changes needed.               |
+============================================================+

Guidelines for writing good descriptions:
- Describe the CONCEPT, not keywords
- Vary the phrasing (FR + EN for the multilingual model)
- Include business variants (e.g. "invoice" + "credit note" + "debit note")
- 4-8 descriptions per class = good stability/coverage trade-off
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types.

    Extensible: add a member here + a RoutePrototype in ROUTES.
    """

    FACTURE = "facture"
    CONTRAT = "contrat"
    RAPPORT = "rapport"
    COURRIER = "courrier"
    TECHNIQUE = "technique"
    RH = "ressources_humaines"
    FORMULAIRE = "formulaire"
    HORS_SUJET = "hors_sujet"
    INCONNU = "inconnu"

    @classmethod
    def classifiable_types(cls) -> list["DocumentType"]:
        """Return types that have a semantic route (excludes INCONNU)."""
        return [t for t in cls if t != cls.INCONNU]


@dataclass(frozen=True)
class RoutePrototype:
    """Semantic prototype for a document type.

    Attributes:
        document_type: The associated type.
        descriptions: Natural language descriptions of the concept.
            The E5 model encodes them with the "passage:" prefix.
            The L2-normalized mean of these embeddings = class centroid.
    """

    document_type: DocumentType
    descriptions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.descriptions) < 2:
            raise ValueError(
                f"Route '{self.document_type.value}' requires at least 2 descriptions, "
                f"got {len(self.descriptions)}."
            )


# ══════════════════════════════════════════════════════════════════════
#  SEMANTIC ROUTES
#
#  Each description captures a FACET of the document concept.
#  The centroid (mean of embeddings) represents the semantic
#  "center of gravity" of the class in vector space.
# ══════════════════════════════════════════════════════════════════════

ROUTES: list[RoutePrototype] = [
    # ── INVOICES ──────────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.FACTURE,
        descriptions=(
            "Document commercial indiquant un montant à payer avec des lignes de facturation détaillées",
            "Invoice with line items, unit prices, total amount due, payment terms and VAT breakdown",
            "Bordereau de paiement avec références client, numéro de facture et détail des prestations",
            "Relevé de charges avec échéances de règlement, TVA et conditions de paiement",
            "Avoir ou note de crédit annulant partiellement une facture précédente",
            "Bon de commande avec prix unitaires, quantités et total HT/TTC",
        ),
    ),
    # ── CONTRACTS ──────────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.CONTRAT,
        descriptions=(
            "Accord juridique entre parties définissant des obligations contractuelles mutuelles",
            "Legal agreement with clauses, terms and conditions, signatures and binding commitments",
            "Convention de partenariat avec conditions générales et particulières signées par les parties",
            "Avenant contractuel modifiant les termes d'un contrat ou engagement existant",
            "Protocole d'accord cadre fixant les règles de collaboration entre organisations",
            "Service level agreement defining performance metrics, penalties and escalation procedures",
        ),
    ),
    # ── REPORTS ──────────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.RAPPORT,
        descriptions=(
            "Document d'analyse présentant des résultats chiffrés, métriques et recommandations",
            "Technical or business report with executive summary, findings, data analysis and conclusions",
            "Étude détaillée avec méthodologie, graphiques, résultats quantitatifs et préconisations",
            "Compte-rendu structuré d'activité avec indicateurs de performance et plan d'action",
            "Audit report presenting compliance findings, risk assessment and remediation steps",
        ),
    ),
    # ── CORRESPONDENCE ──────────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.COURRIER,
        descriptions=(
            "Correspondance formelle adressée à un destinataire avec objet, date et signature",
            "Formal letter with sender address, recipient, subject line, greeting and closing",
            "Notification officielle ou mise en demeure adressée à une personne ou organisation",
            "Lettre de réclamation ou de demande d'information envoyée par courrier postal ou email",
            "Courrier administratif avec références, tampons et accusé de réception",
        ),
    ),
    # ── TECHNICAL ─────────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.TECHNIQUE,
        descriptions=(
            "Documentation technique avec spécifications système, schémas d'architecture et procédures",
            "Technical specification with API references, architecture diagrams and deployment guides",
            "Manuel d'utilisation avec instructions pas-à-pas, captures d'écran et troubleshooting",
            "Cahier des charges fonctionnel et technique détaillant les exigences d'un projet",
            "Release notes or changelog documenting software version updates and bug fixes",
        ),
    ),
    # ── HUMAN RESOURCES ───────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.RH,
        descriptions=(
            "Document relatif à la gestion du personnel : contrat de travail, fiche de paie, attestation",
            "Employment contract, payslip, leave request or performance review document",
            "Bulletin de salaire avec détail des cotisations sociales, net à payer et cumuls annuels",
            "Demande de congés, attestation employeur ou certificat de travail",
            "Organigramme, fiche de poste ou plan de formation des collaborateurs",
        ),
    ),
    # ── FORMS ───────────────────────────────────────────────
    RoutePrototype(
        document_type=DocumentType.FORMULAIRE,
        descriptions=(
            "Formulaire à remplir avec des champs vides, cases à cocher et zones de signature",
            "Structured form with blank fields, checkboxes, dropdown selections and submission instructions",
            "Questionnaire ou enquête avec questions numérotées et espaces de réponse",
            "Cerfa ou formulaire administratif officiel avec numéro de référence réglementaire",
        ),
    ),
    # ── NOISE / OUT OF DOMAIN ──────────────────────────────
    RoutePrototype(
        document_type=DocumentType.HORS_SUJET,
        descriptions=(
            "Texte aléatoire sans signification sémantique, suite de mots incohérente ou remplissage (lorem ipsum)",
            "Random gibberish text with no semantic meaning, placeholder content or scrambled letters",
            "Recette de cuisine, poème, texte littéraire ou personnel sans lien avec l'entreprise",
            "Cooking recipe, poetry, fiction or personal blog post unrelated to business context",
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt",
        ),
    ),
]


def get_routes() -> list[RoutePrototype]:
    """Single access point for routes. Allows future dynamic loading."""
    return ROUTES
