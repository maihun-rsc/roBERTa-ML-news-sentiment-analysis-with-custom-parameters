"""
test_module2.py
───────────────
Unit tests for Module 2 — Preprocessing.

Uses spaCy's blank('en') pipeline + manually-added components where a
trained model isn't available in the test environment (no network access
to download en_core_web_sm/trf in CI/sandbox). The actual production NER
quality is achieved by load_ner_model() at runtime, which IS network-
dependent and tested separately as an integration concern, not here.

Run:
    python test_module2.py
    python test_module2.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ════════════════════════════════════════════════════════════════
#  Test: cleaner.py
# ════════════════════════════════════════════════════════════════
class TestCleaner(unittest.TestCase):

    def test_strip_html(self):
        from preprocessing.cleaner import strip_html
        result = strip_html("<p>Hello <b>world</b></p>")
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_strip_html_entities(self):
        from preprocessing.cleaner import strip_html
        result = strip_html("Tom &amp; Jerry &#39;s show")
        self.assertNotIn("&amp;", result)
        self.assertNotIn("&#39;", result)

    def test_normalize_unicode_smart_quotes(self):
        from preprocessing.cleaner import normalize_unicode
        result = normalize_unicode("\u2018Hello\u2019 \u201cworld\u201d")
        self.assertEqual(result, "'Hello' \"world\"")

    def test_normalize_unicode_dashes(self):
        from preprocessing.cleaner import normalize_unicode
        result = normalize_unicode("2020\u20132024 \u2014 a period")
        self.assertIn("2020-2024", result)
        self.assertIn("- a period", result)

    def test_normalize_unicode_preserves_accents(self):
        """Proper noun accents must survive — these are not typographic noise."""
        from preprocessing.cleaner import normalize_unicode
        result = normalize_unicode("President Erdoğan visited Şanlıurfa")
        self.assertIn("Erdoğan", result)
        self.assertIn("Şanlıurfa", result)

    def test_remove_boilerplate_advertisement(self):
        from preprocessing.cleaner import remove_boilerplate
        text = "Real content here.\nADVERTISEMENT\nMore real content."
        result = remove_boilerplate(text)
        self.assertNotIn("ADVERTISEMENT", result.upper().replace("\n", " "))

    def test_remove_boilerplate_subscribe(self):
        from preprocessing.cleaner import remove_boilerplate
        text = "Important news here.\nSubscribe to our newsletter for updates.\nMore news."
        result = remove_boilerplate(text)
        self.assertNotIn("Subscribe to our newsletter", result)

    def test_remove_boilerplate_preserves_real_content(self):
        """Most critical test: boilerplate removal must not eat real sentences."""
        from preprocessing.cleaner import remove_boilerplate
        text = "The minister announced a new policy on Tuesday regarding the economy."
        result = remove_boilerplate(text)
        self.assertEqual(result.strip(), text.strip())

    def test_remove_urls(self):
        from preprocessing.cleaner import remove_urls
        text = "Read more at https://example.com/article123 for details."
        result = remove_urls(text)
        self.assertNotIn("https://", result)
        self.assertIn("Read more", result)
        self.assertIn("for details", result)

    def test_collapse_whitespace(self):
        from preprocessing.cleaner import collapse_whitespace
        text = "Too    many     spaces.\n\n\n\nToo many newlines."
        result = collapse_whitespace(text)
        self.assertNotIn("    ", result)
        self.assertNotIn("\n\n\n", result)

    def test_deduplicate_paragraphs(self):
        from preprocessing.cleaner import deduplicate_paragraphs
        text = "First paragraph.\n\nSecond paragraph.\n\nFirst paragraph."
        result = deduplicate_paragraphs(text)
        self.assertEqual(result.count("First paragraph."), 1)
        self.assertIn("Second paragraph.", result)

    def test_clean_text_preserves_negation(self):
        """CRITICAL: negation words must never be stripped — they carry
        essential framing meaning per the project's core preprocessing rule."""
        from preprocessing.cleaner import clean_text
        text = "The minister did not deny the allegations, despite pressure."
        result = clean_text(text)
        self.assertIn("not", result)
        self.assertIn("despite", result)

    def test_clean_text_preserves_modality_words(self):
        from preprocessing.cleaner import clean_text
        text = "Officials allegedly ignored warnings, reportedly due to budget cuts."
        result = clean_text(text)
        self.assertIn("allegedly", result)
        self.assertIn("reportedly", result)

    def test_clean_text_does_not_lowercase(self):
        from preprocessing.cleaner import clean_text
        text = "Boris Johnson met the President of France."
        result = clean_text(text)
        self.assertIn("Boris Johnson", result)
        self.assertIn("President", result)

    def test_clean_text_empty_input(self):
        from preprocessing.cleaner import clean_text
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_clean_title(self):
        from preprocessing.cleaner import clean_title
        result = clean_title("<h1>Breaking: \u201cMajor\u201d News</h1>")
        self.assertNotIn("<h1>", result)
        self.assertIn('"Major"', result)


# ════════════════════════════════════════════════════════════════
#  Test: ner_pipeline.py (using spaCy blank + manual entity injection)
# ════════════════════════════════════════════════════════════════
class TestNERPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Build a minimal spaCy pipeline using only components that don't
        require a downloaded model: tokenizer + rule-based sentencizer +
        a manually-populated EntityRuler. This validates our pipeline
        LOGIC (graph building, frequency counting, span extraction)
        independent of NER model accuracy, which is a separate concern
        tested at the model-loading level.
        """
        import spacy

        cls.nlp = spacy.blank("en")
        cls.nlp.add_pipe("sentencizer")

        ruler = cls.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "PERSON", "pattern": "Boris Johnson"},
            {"label": "GPE",    "pattern": "London"},
            {"label": "GPE",    "pattern": "France"},
            {"label": "ORG",    "pattern": "United Nations"},
        ]
        ruler.add_patterns(patterns)

    def test_process_document_basic(self):
        from preprocessing.ner_pipeline import process_document
        text = "Boris Johnson visited London yesterday to discuss trade with France."
        doc = process_document(self.nlp, text)

        self.assertGreater(len(doc.tokens), 0)
        self.assertEqual(len(doc.tokens), len(doc.pos_tags))
        self.assertEqual(len(doc.tokens), len(doc.dep_labels))

    def test_process_document_extracts_entities(self):
        from preprocessing.ner_pipeline import process_document
        text = "Boris Johnson visited London yesterday."
        doc = process_document(self.nlp, text)

        entity_texts = [e.text for e in doc.entities]
        self.assertIn("Boris Johnson", entity_texts)
        self.assertIn("London", entity_texts)

    def test_process_document_filters_unwanted_types(self):
        """CARDINAL, DATE, MONEY etc. should be dropped — not framing targets."""
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([
            {"label": "DATE",     "pattern": "yesterday"},
            {"label": "CARDINAL", "pattern": "five"},
            {"label": "PERSON",   "pattern": "Boris Johnson"},
        ])

        from preprocessing.ner_pipeline import process_document
        doc = process_document(nlp, "Boris Johnson met five officials yesterday.")

        labels = {e.label for e in doc.entities}
        self.assertIn("PERSON", labels)
        self.assertNotIn("DATE", labels)
        self.assertNotIn("CARDINAL", labels)

    def test_process_document_empty_text(self):
        from preprocessing.ner_pipeline import process_document
        doc = process_document(self.nlp, "")
        self.assertEqual(doc.tokens, [])
        self.assertEqual(doc.entities, [])

    def test_entity_frequency_counting(self):
        from preprocessing.ner_pipeline import process_document
        text = "Boris Johnson spoke today. Boris Johnson later met officials in London."
        doc = process_document(self.nlp, text)
        self.assertEqual(doc.entity_freq.get("Boris Johnson"), 2)

    def test_get_primary_entities_ranks_by_frequency(self):
        from preprocessing.ner_pipeline import process_document, get_primary_entities
        text = (
            "Boris Johnson spoke today. Boris Johnson met officials. "
            "London hosted the event."
        )
        doc = process_document(self.nlp, text)
        top = get_primary_entities(doc, top_k=2)
        self.assertEqual(top[0], "Boris Johnson")  # mentioned twice, ranks first

    def test_get_primary_entities_empty(self):
        from preprocessing.ner_pipeline import process_document, get_primary_entities
        doc = process_document(self.nlp, "")
        self.assertEqual(get_primary_entities(doc), [])

    def test_sentence_boundaries_populated(self):
        from preprocessing.ner_pipeline import process_document
        text = "First sentence here. Second sentence follows. Third one too."
        doc = process_document(self.nlp, text)
        self.assertEqual(len(doc.sent_boundaries), 3)

    def test_entity_sent_idx_assignment(self):
        from preprocessing.ner_pipeline import process_document
        text = "Nothing interesting here. Boris Johnson appears in this sentence."
        doc = process_document(self.nlp, text)
        boris = next(e for e in doc.entities if e.text == "Boris Johnson")
        self.assertEqual(boris.sent_idx, 1)  # second sentence, 0-indexed

    def test_batch_process_matches_individual(self):
        from preprocessing.ner_pipeline import process_document, batch_process
        texts = [
            "Boris Johnson visited London.",
            "France hosted the United Nations summit.",
        ]
        individual = [process_document(self.nlp, t) for t in texts]
        batched = batch_process(self.nlp, texts)

        self.assertEqual(len(individual), len(batched))
        for ind, bat in zip(individual, batched):
            self.assertEqual(len(ind.tokens), len(bat.tokens))
            self.assertEqual(
                sorted(e.text for e in ind.entities),
                sorted(e.text for e in bat.entities),
            )

    def test_batch_process_empty_list(self):
        from preprocessing.ner_pipeline import batch_process
        result = batch_process(self.nlp, [])
        self.assertEqual(result, [])

    def test_batch_process_handles_empty_string_in_batch(self):
        from preprocessing.ner_pipeline import batch_process
        texts = ["Boris Johnson visited London.", "", "France hosted summit."]
        results = batch_process(self.nlp, texts)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1].tokens, [])  # empty string → empty doc


# ════════════════════════════════════════════════════════════════
#  Test: proximity_scorer.py
# ════════════════════════════════════════════════════════════════
class TestProximityScorer(unittest.TestCase):

    def test_build_dependency_graph_simple(self):
        from preprocessing.proximity_scorer import build_dependency_graph
        # token 1 is root (head=1), token 0's head is 1, token 2's head is 1
        head_indices = [1, 1, 1]
        graph = build_dependency_graph(head_indices)
        self.assertIn(1, graph[0])
        self.assertIn(0, graph[1])
        self.assertIn(2, graph[1])

    def test_shortest_path_same_node(self):
        from preprocessing.proximity_scorer import build_dependency_graph, shortest_path_length
        graph = build_dependency_graph([1, 1, 1])
        self.assertEqual(shortest_path_length(graph, 0, 0), 0)

    def test_shortest_path_direct_neighbor(self):
        from preprocessing.proximity_scorer import build_dependency_graph, shortest_path_length
        graph = build_dependency_graph([1, 1, 1])  # 0-1-2 star with 1 as center
        self.assertEqual(shortest_path_length(graph, 0, 1), 1)

    def test_shortest_path_two_hops(self):
        from preprocessing.proximity_scorer import build_dependency_graph, shortest_path_length
        graph = build_dependency_graph([1, 1, 1])  # 0 and 2 both connect via 1
        self.assertEqual(shortest_path_length(graph, 0, 2), 2)

    def test_shortest_path_unreachable_returns_max(self):
        from preprocessing.proximity_scorer import build_dependency_graph, shortest_path_length
        graph = {0: [], 1: []}  # disconnected
        result = shortest_path_length(graph, 0, 1, max_distance=10)
        self.assertEqual(result, 10)

    def test_compute_proximity_scores_decreasing_with_distance(self):
        from preprocessing.ner_pipeline import ProcessedDoc, EntitySpan
        from preprocessing.proximity_scorer import compute_proximity_scores

        # Chain: 0 -> 1 -> 2 -> 3 (each token's head is the next one, last is root)
        doc = ProcessedDoc(
            tokens=["The", "minister", "strongly", "denied"],
            head_indices=[1, 3, 3, 3],  # "denied" (idx 3) is root
        )
        entity = EntitySpan(text="minister", label="PERSON", start=4, end=12, token_idx=1)

        scores = compute_proximity_scores(doc, entity, decay="inverse")

        # Token 1 (the entity itself) should score 1.0 (distance 0)
        self.assertAlmostEqual(scores[1], 1.0)
        # Scores should generally decrease as dependency distance increases
        self.assertGreaterEqual(scores[1], scores[0])

    def test_compute_proximity_scores_empty_doc(self):
        from preprocessing.ner_pipeline import ProcessedDoc, EntitySpan
        from preprocessing.proximity_scorer import compute_proximity_scores
        doc = ProcessedDoc(tokens=[], head_indices=[])
        entity = EntitySpan(text="x", label="PERSON", start=0, end=1, token_idx=0)
        self.assertEqual(compute_proximity_scores(doc, entity), [])

    def test_compute_all_entity_proximities(self):
        from preprocessing.ner_pipeline import ProcessedDoc, EntitySpan
        from preprocessing.proximity_scorer import compute_all_entity_proximities

        doc = ProcessedDoc(
            tokens=["Boris", "met", "Macron"],
            head_indices=[1, 1, 1],
            entities=[
                EntitySpan(text="Boris", label="PERSON", start=0, end=5, token_idx=0),
                EntitySpan(text="Macron", label="PERSON", start=10, end=16, token_idx=2),
            ],
        )
        result = compute_all_entity_proximities(doc)
        self.assertIn("Boris", result)
        self.assertIn("Macron", result)
        self.assertEqual(len(result["Boris"]), 3)

    def test_get_entity_context_window(self):
        from preprocessing.ner_pipeline import ProcessedDoc, EntitySpan
        from preprocessing.proximity_scorer import get_entity_context_window

        doc = ProcessedDoc(
            tokens=list(range(20)),  # 20 dummy tokens
            sent_boundaries=[(0, 5), (5, 10), (10, 15), (15, 20)],
        )
        entity = EntitySpan(text="x", label="PERSON", start=0, end=1, sent_idx=2, token_idx=12)

        start, end = get_entity_context_window(doc, entity, window_sentences=1)
        # Sentence 2 ± 1 = sentences 1,2,3 → token range (5, 20)
        self.assertEqual(start, 5)
        self.assertEqual(end, 20)

    def test_get_entity_context_window_clamps_at_boundaries(self):
        from preprocessing.ner_pipeline import ProcessedDoc, EntitySpan
        from preprocessing.proximity_scorer import get_entity_context_window

        doc = ProcessedDoc(
            tokens=list(range(10)),
            sent_boundaries=[(0, 5), (5, 10)],
        )
        entity = EntitySpan(text="x", label="PERSON", start=0, end=1, sent_idx=0, token_idx=2)
        start, end = get_entity_context_window(doc, entity, window_sentences=5)
        self.assertEqual(start, 0)
        self.assertEqual(end, 10)


# ════════════════════════════════════════════════════════════════
#  Test: asr_cleaner.py
# ════════════════════════════════════════════════════════════════
class TestASRCleaner(unittest.TestCase):

    def test_remove_disfluencies_filler_words(self):
        from preprocessing.asr_cleaner import remove_disfluencies
        text = "So, um, the minister, uh, announced a new policy today."
        result = remove_disfluencies(text)
        self.assertNotIn(" um", result.lower())
        self.assertNotIn(" uh", result.lower())
        self.assertIn("minister", result)
        self.assertIn("announced", result)

    def test_remove_disfluencies_preserves_hedging_words(self):
        """CRITICAL: 'allegedly'/'reportedly' must survive — not disfluencies."""
        from preprocessing.asr_cleaner import remove_disfluencies
        text = "Officials allegedly ignored the warning, reportedly due to budget cuts."
        result = remove_disfluencies(text)
        self.assertIn("allegedly", result)
        self.assertIn("reportedly", result)

    def test_remove_disfluencies_bracket_tags(self):
        from preprocessing.asr_cleaner import remove_disfluencies
        text = "The crowd reacted [applause] as the minister spoke [crosstalk]."
        result = remove_disfluencies(text)
        self.assertNotIn("[applause]", result)
        self.assertNotIn("[crosstalk]", result)

    def test_remove_disfluencies_repeated_words(self):
        from preprocessing.asr_cleaner import remove_disfluencies
        text = "The the minister announced announced a policy."
        result = remove_disfluencies(text)
        self.assertNotIn("the the", result.lower())
        self.assertNotIn("announced announced", result.lower())

    def test_correct_entity_misrecognitions(self):
        from preprocessing.asr_cleaner import correct_entity_misrecognitions
        text = "President by den spoke with Zelinski about the conflict."
        result = correct_entity_misrecognitions(text)
        self.assertIn("Biden", result)
        self.assertIn("Zelenskyy", result)

    def test_filter_low_confidence_segments(self):
        from preprocessing.asr_cleaner import TranscriptSegment, filter_low_confidence_segments
        segments = [
            TranscriptSegment(text="Clear speech here", start=0, end=2, avg_logprob=-0.3),
            TranscriptSegment(text="garbled mumbling", start=2, end=4, avg_logprob=-1.8),
            TranscriptSegment(text="More clear speech", start=4, end=6, avg_logprob=-0.5),
        ]
        kept, masked = filter_low_confidence_segments(segments, logprob_threshold=-1.0)
        self.assertEqual(len(kept), 2)
        self.assertEqual(masked, 1)

    def test_segments_to_sentences_adds_terminal_punctuation(self):
        from preprocessing.asr_cleaner import TranscriptSegment, segments_to_sentences
        segments = [
            TranscriptSegment(text="The minister spoke today", start=0, end=2, avg_logprob=-0.3),
            TranscriptSegment(text="Officials reacted swiftly", start=2, end=4, avg_logprob=-0.4),
        ]
        result = segments_to_sentences(segments)
        self.assertIn("today.", result)
        self.assertIn("swiftly.", result)

    def test_clean_transcript_full_pipeline(self):
        from preprocessing.asr_cleaner import TranscriptSegment, clean_transcript
        segments = [
            TranscriptSegment(text="So, um, the minister, uh, spoke today", start=0, end=3, avg_logprob=-0.4),
            TranscriptSegment(text="completely garbled noise", start=3, end=5, avg_logprob=-2.0),
            TranscriptSegment(text="President by den responded", start=5, end=8, avg_logprob=-0.3),
        ]
        result = clean_transcript(segments, logprob_threshold=-1.0)

        self.assertEqual(result.total_segments, 3)
        self.assertEqual(result.masked_segments, 1)
        self.assertIn("Biden", result.text)
        self.assertNotIn(" um", result.text.lower())

    def test_whisper_result_to_segments_adapter(self):
        from preprocessing.asr_cleaner import whisper_result_to_segments
        fake_whisper_output = {
            "text": "full transcript",
            "segments": [
                {"text": "First segment", "start": 0.0, "end": 2.5, "avg_logprob": -0.3},
                {"text": "Second segment", "start": 2.5, "end": 5.0, "avg_logprob": -0.5},
            ],
        }
        segments = whisper_result_to_segments(fake_whisper_output)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "First segment")
        self.assertEqual(segments[1].avg_logprob, -0.5)


# ════════════════════════════════════════════════════════════════
#  Test: __init__.py orchestrator (preprocess_article, run_preprocessing)
# ════════════════════════════════════════════════════════════════
class TestPreprocessOrchestrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import spacy
        cls.nlp = spacy.blank("en")
        cls.nlp.add_pipe("sentencizer")
        ruler = cls.nlp.add_pipe("entity_ruler")
        ruler.add_patterns([
            {"label": "PERSON", "pattern": "Rishi Sunak"},
            {"label": "GPE",    "pattern": "London"},
        ])

    def _make_article(self):
        from data_collection.schema import Article, make_article_id
        body = "<p>Rishi Sunak visited London today to discuss the economy.</p>"
        return Article(
            article_id=make_article_id(body),
            source="bbc",
            title="<b>PM Visits London</b>",
            body=body,
            url="https://bbc.com/test",
            date="2024-01-01T00:00:00+00:00",
            topic="politics",
        )

    def test_preprocess_article_populates_fields(self):
        from preprocessing import preprocess_article
        article = self._make_article()
        result = preprocess_article(self.nlp, article)

        self.assertNotIn("<p>", result.clean_body)
        self.assertNotIn("<b>", result.title)
        self.assertIn("Rishi Sunak", result.entities)
        self.assertGreater(len(result.tokens), 0)
        self.assertGreater(len(result.entity_spans), 0)

    def test_preprocess_article_entity_spans_have_required_keys(self):
        from preprocessing import preprocess_article
        article = self._make_article()
        result = preprocess_article(self.nlp, article)

        for span in result.entity_spans:
            self.assertIn("text", span)
            self.assertIn("label", span)
            self.assertIn("start", span)
            self.assertIn("end", span)

    def test_preprocess_article_empty_body(self):
        from preprocessing import preprocess_article
        from data_collection.schema import Article
        article = Article(
            article_id="x", source="bbc", title="Title",
            body="<p></p>", url="https://x.com",
            date="2024-01-01T00:00:00+00:00", topic="general",
        )
        result = preprocess_article(self.nlp, article)
        self.assertEqual(result.entities, [])


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
