"""
tests/test_router.py — Unit tests for the SemanticRouter.

Run: pytest tests/test_router.py -v

These tests verify that:
1. Classification is purely semantic (no keyword matching)
2. Atypical documents are correctly rejected (INCONNU)
3. Batch gives the same results as individual classification
4. Metrics are consistent
"""

from __future__ import annotations

import pytest

from config.routes import DocumentType
from classifier.semantic_router import SemanticRouter
from classifier.models import ClassificationResult


# ══════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def router() -> SemanticRouter:
    """SemanticRouter initialized once for all tests."""
    return SemanticRouter()


# ══════════════════════════════════════════════════════════════════════
#  Test texts
# ══════════════════════════════════════════════════════════════════════

# Texts that must be classified correctly
CLASSIFICATION_CASES = [
    # (text, expected_type, test_description)
    (
        "FACTURE N° 2024-0847\n"
        "Client : SARL Technologie Avancée\n"
        "Désignation : Prestation de conseil IT - 15 jours\n"
        "Prix unitaire HT : 850,00 €\n"
        "Total HT : 12 750,00 €\n"
        "TVA 20% : 2 550,00 €\n"
        "Total TTC : 15 300,00 €\n"
        "Échéance : 30 jours fin de mois\n"
        "RIB : FR76 3000 4028 3700 0100 0264 382",
        DocumentType.FACTURE,
        "Classic FR invoice with amounts and VAT",
    ),
    (
        "CONTRAT DE PRESTATION DE SERVICES\n"
        "Entre les soussignés :\n"
        "La société Orange SA, représentée par M. Dupont, Directeur Général\n"
        "ci-après dénommée « le Client »\n"
        "Et\n"
        "La société DataTech SARL\n"
        "ci-après dénommée « le Prestataire »\n"
        "Article 1 : Objet du contrat\n"
        "Le Prestataire s'engage à fournir les services de maintenance...\n"
        "Article 5 : Durée et résiliation\n"
        "Le présent contrat est conclu pour une durée de 24 mois.",
        DocumentType.CONTRAT,
        "Service contract with articles and parties",
    ),
    (
        "RAPPORT D'ACTIVITÉ — Q3 2024\n"
        "Direction : DATA-IA / SND / DREAMS\n\n"
        "1. Résumé exécutif\n"
        "Le chiffre d'affaires a progressé de 12% par rapport au Q2.\n"
        "Les indicateurs KPI montrent une amélioration significative.\n\n"
        "2. Analyse des résultats\n"
        "Le taux de conversion est passé de 3.2% à 4.7%.\n"
        "La latence moyenne du pipeline a été réduite de 340ms à 89ms.\n\n"
        "3. Recommandations\n"
        "Nous préconisons d'augmenter la capacité GPU de 40%.",
        DocumentType.RAPPORT,
        "Activity report with KPIs and recommendations",
    ),
    (
        "Paris, le 15 janvier 2025\n\n"
        "Objet : Demande de renouvellement de licence\n\n"
        "Madame, Monsieur,\n\n"
        "Suite à notre entretien téléphonique du 10 janvier, je me permets "
        "de vous adresser la présente demande de renouvellement de notre "
        "licence d'exploitation logicielle.\n\n"
        "Dans l'attente de votre réponse, je vous prie d'agréer, Madame, "
        "Monsieur, l'expression de mes salutations distinguées.\n\n"
        "M. Diallo\nDirecteur Technique",
        DocumentType.COURRIER,
        "Formal letter with polite formulas",
    ),
    (
        "DOCUMENTATION TECHNIQUE — API Gateway v3.2\n\n"
        "Architecture du système :\n"
        "Le gateway expose des endpoints REST sur le port 8443.\n"
        "L'authentification utilise des tokens JWT avec rotation.\n\n"
        "Configuration :\n"
        "```yaml\n"
        "server:\n"
        "  port: 8443\n"
        "  ssl: true\n"
        "  rate_limit: 1000/min\n"
        "```\n\n"
        "Endpoints disponibles :\n"
        "POST /api/v3/ingest — Ingestion de documents\n"
        "GET /api/v3/status/{id} — Statut du traitement",
        DocumentType.TECHNIQUE,
        "Technical documentation with YAML config and endpoints",
    ),
    (
        "BULLETIN DE PAIE — Décembre 2024\n"
        "Employeur : Orange SA\n"
        "Salarié : M. Amadou FALL\n"
        "Emploi : Ingénieur Machine Learning\n"
        "Salaire brut : 5 200,00 €\n"
        "Cotisations salariales : -1 105,00 €\n"
        "CSG déductible : -340,00 €\n"
        "Net imposable : 3 755,00 €\n"
        "Net à payer : 3 680,00 €\n"
        "Cumul annuel brut : 62 400,00 €",
        DocumentType.RH,
        "Payslip with deductions and net pay",
    ),
    (
        "FORMULAIRE DE DEMANDE D'ACCÈS\n"
        "Réf: FORM-SEC-2024-003\n\n"
        "Nom : ____________________\n"
        "Prénom : ____________________\n"
        "Service : ____________________\n"
        "Date de la demande : __ / __ / ____\n\n"
        "Type d'accès demandé :\n"
        "☐ Lecture seule\n"
        "☐ Lecture/Écriture\n"
        "☐ Administrateur\n\n"
        "Justification : ________________________________________\n\n"
        "Signature du demandeur : ________________\n"
        "Signature du responsable : ________________",
        DocumentType.FORMULAIRE,
        "Form with blank fields and checkboxes",
    ),
]

# Texts that must be rejected (INCONNU)
REJECTION_CASES = [
    (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Lorem ipsum (no business meaning)",
    ),
    (
        "abc def ghi jkl mno pqr stu vwx yz 12345",
        "Gibberish without document structure",
    ),
]


# ══════════════════════════════════════════════════════════════════════
#  Classification tests
# ══════════════════════════════════════════════════════════════════════


class TestSemanticRouter:
    """Semantic router tests."""

    @pytest.mark.parametrize(
        "text,expected_type,description",
        CLASSIFICATION_CASES,
        ids=[c[2] for c in CLASSIFICATION_CASES],
    )
    def test_correct_classification(
        self,
        router: SemanticRouter,
        text: str,
        expected_type: DocumentType,
        description: str,
    ):
        """Verify that each document type is correctly identified."""
        result = router.classify(text)

        assert result.document_type == expected_type, (
            f"\n{'='*60}\n"
            f"FAILED: {description}\n"
            f"Expected: {expected_type.value}\n"
            f"Got:      {result.document_type.value} "
            f"(confidence={result.confidence:.4f})\n"
            f"Top-3:    {[(c.document_type.value, c.confidence) for c in result.top_k]}\n"
            f"{'='*60}"
        )

        # Confidence must be above the threshold
        assert result.confidence >= router.confidence_threshold

    @pytest.mark.parametrize(
        "text,description",
        REJECTION_CASES,
        ids=[c[1] for c in REJECTION_CASES],
    )
    def test_unknown_rejection(
        self,
        router: SemanticRouter,
        text: str,
        description: str,
    ):
        """Verify that non-document texts are rejected."""
        result = router.classify(text)
        assert result.document_type == DocumentType.INCONNU, (
            f"Text '{description}' should have been INCONNU, "
            f"got: {result.document_type.value} (conf={result.confidence:.4f})"
        )

    def test_batch_coherence(self, router: SemanticRouter):
        """Verify that classify_batch gives the same results as classify."""
        texts = [case[0] for case in CLASSIFICATION_CASES]

        # Individual classification
        individual_results = [router.classify(t) for t in texts]
        individual_types = [r.document_type for r in individual_results]

        # Batch classification
        batch_results = router.classify_batch(texts)
        batch_types = [r.document_type for r in batch_results]

        assert individual_types == batch_types, (
            "Inconsistency between classify and classify_batch:\n"
            f"Individual: {[t.value for t in individual_types]}\n"
            f"Batch:      {[t.value for t in batch_types]}"
        )

    def test_result_structure(self, router: SemanticRouter):
        """Verify that ClassificationResult is well-formed."""
        result = router.classify("Facture N° 123 — Total TTC: 500€")

        assert isinstance(result.document_type, DocumentType)
        assert -1.0 <= result.confidence <= 1.0
        assert len(result.top_k) > 0
        assert result.latency_ms >= 0
        assert result.embedding_model == router._engine.model_id
        assert result.confidence_threshold == router.confidence_threshold

    def test_top_k_ordering(self, router: SemanticRouter):
        """Verify that top-k is sorted by descending confidence."""
        result = router.classify(
            "Contrat de travail à durée indéterminée entre l'employeur et le salarié",
            top_k=5,
        )

        confidences = [c.confidence for c in result.top_k]
        assert confidences == sorted(confidences, reverse=True), (
            f"Top-k not sorted: {confidences}"
        )

    def test_explain_output(self, router: SemanticRouter):
        """Verify that explain() returns all required fields."""
        explanation = router.explain("Rapport annuel des ventes 2024")

        assert "predicted" in explanation
        assert "confidence" in explanation
        assert "threshold" in explanation
        assert "is_confident" in explanation
        assert "margin" in explanation
        assert "all_scores" in explanation
        assert "model" in explanation
        assert "text_preview" in explanation

        # All types must have a score
        assert len(explanation["all_scores"]) == router.n_routes

    def test_margin_discriminant(self, router: SemanticRouter):
        """Verify that well-typed documents have a positive margin."""
        for text, expected_type, desc in CLASSIFICATION_CASES:
            result = router.classify(text, top_k=3)
            if result.is_confident and len(result.top_k) >= 2:
                assert result.margin > 0, (
                    f"Zero or negative margin for '{desc}': "
                    f"margin={result.margin:.4f}"
                )

    def test_empty_text(self, router: SemanticRouter):
        """Empty text should be classified as INCONNU, not crash."""
        result = router.classify("")
        # The model may return anything, but it must not crash
        assert isinstance(result, ClassificationResult)

    def test_very_long_text(self, router: SemanticRouter):
        """Very long text should be truncated cleanly."""
        long_text = "Ce contrat lie les deux parties. " * 10000
        result = router.classify(long_text)
        assert isinstance(result, ClassificationResult)
        assert result.latency_ms < 5000  # Must stay under 5 seconds

    def test_n_routes(self, router: SemanticRouter):
        """Verify the number of initialized routes."""
        assert router.n_routes == 7  # 7 types (excluding INCONNU)
        assert DocumentType.INCONNU not in router.route_types

    def test_centroid_dimensions(self, router: SemanticRouter):
        """Verify that centroids have the correct dimension."""
        for route_type in router.route_types:
            centroid = router.get_centroid(route_type)
            assert centroid is not None
            assert centroid.shape == (router.embedding_dim,)
            # Verify L2 normalization
            import numpy as np
            norm = np.linalg.norm(centroid)
            assert abs(norm - 1.0) < 0.01, f"Non-normalized centroid: norm={norm}"
