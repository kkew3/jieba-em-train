from argparse import ArgumentParser

from bs4 import BeautifulSoup


def main():
    parser = ArgumentParser(description="Parse msra xml file into plaintext.")
    parser.add_argument("msra_xml_file")
    parser.add_argument("output_file", help="Write output to this file")
    args = parser.parse_args()

    with open(args.msra_xml_file, "rb") as infile:
        soup = BeautifulSoup(infile, "lxml-xml")
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for sent in soup.find_all("SENTENCE"):
            segmented_text = [token.text for token in sent.find_all("w")]
            outfile.write(" ".join(segmented_text) + "\n")
