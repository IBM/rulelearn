from unittest import TestCase

import rulelearn.trxf.classifier.ruleset_classifier as classifier
from rulelearn.trxf.classifier.ruleset_classifier import RuleSetClassifier
from rulelearn.trxf.pmml_export.models import RuleSetModel, SimplePMMLRuleSetModel
from rulelearn.trxf.pmml_export.reader.trxf_reader import TrxfReader
from tests.trxf.utilities import create_test_ruleset, DATA_DICTIONARY, TEST_MINING_SCHEMA, \
    TEST_PMML_RULESET, DATA_FRAME


class TestTrxfReader(TestCase):
    def test_read(self):
        reader = TrxfReader(DATA_DICTIONARY)
        test_ruleset = create_test_ruleset()
        test_classifier = RuleSetClassifier([test_ruleset], classifier.RuleSelectionMethod.FIRST_HIT, default_label=0)
        ruleset_model = RuleSetModel(miningSchema=TEST_MINING_SCHEMA, ruleSet=TEST_PMML_RULESET)
        expected = SimplePMMLRuleSetModel(dataDictionary=DATA_DICTIONARY, ruleSetModel=ruleset_model)
        self.assertEqual(reader.read(test_classifier), expected)

    def test_load_data_dictionary(self):
        reader = TrxfReader()
        reader.load_data_dictionary(DATA_FRAME)
        self.assertEqual(reader.data_dictionary, DATA_DICTIONARY)
