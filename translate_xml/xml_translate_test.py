import unittest
from xml_translate import XMLtrans

def read_file(input):
    with open(input) as file:
        return(file.read())

class TestUser(unittest.TestCase):
    def test_reading_untagged(self):
        xml_translation = XMLtrans("tests/input1.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, "This text should not be modified.")

    def test_joinning(self):
        xml_translation = XMLtrans("tests/input2.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, "This text should not be modified. Both text should be joined.")

    def test_fail_to_read_ID(self):
        xml_translation = XMLtrans("tests/input3.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, "This word should not be modified.")

    def test_replace_by_id(self):
        xml_translation = XMLtrans("tests/input4.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, "This MeSHA0001234 should be replaced by MeSHA0001234.")

    def test_several_translations(self):
        xml_translation = XMLtrans("tests/input5.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, "Final test of MeSHA012345 to see if MeSHQ000628. Also testing the joining of different information (should not be translated).")

    def test_annotation_1(self):
        xml_translation = XMLtrans("sample_annotated_xml_mod/10605405_annotated.xml", "")
        xml_translation.translate()
        output_text = xml_translation.output_text()
        self.assertEqual(output_text, read_file("tests/annotation_1_text.xml"))

if __name__ == '__main__':
    unittest.main()