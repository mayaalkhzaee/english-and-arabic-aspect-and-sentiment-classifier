import json
from lxml import etree

def convert_xml_to_jsonl(xml_path: str, output_path: str):
    """
    Converts Laptop_Train_v2.xml into final_dataset.jsonl
    following the structure:
    {
        "id": int,
        "sentence": str,
        "aspect_terms": [
            {"term": str, "polarity": str, "from": int, "to": int},
            ...
        ]
    }
    """

    tree = etree.parse(xml_path)
    root = tree.getroot()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for sentence in root.findall("sentence"):
            sent_id = sentence.get("id")
            text_elem = sentence.find("text")

            # Skip if no text found
            if text_elem is None or not text_elem.text:
                continue

            sentence_text = text_elem.text.strip()

            # Extract aspect terms list
            aspect_terms_list = []
            aspectTerms = sentence.find("aspectTerms")

            if aspectTerms is not None:
                for term in aspectTerms.findall("aspectTerm"):
                    aspect_terms_list.append({
                        "term": term.get("term"),
                        "polarity": term.get("polarity"),
                        "from": int(term.get("from")),
                        "to": int(term.get("to"))
                    })

            # Build final record
            record = {
                "id": int(sent_id),
                "sentence": sentence_text,
                "aspect_terms": aspect_terms_list
            }

            # Write one JSON object per line
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✔ Successfully written dataset to {output_path}")


# =============================
# Run the conversion
# =============================

convert_xml_to_jsonl("Laptop_Train_v2.xml", "final_dataset.jsonl")
