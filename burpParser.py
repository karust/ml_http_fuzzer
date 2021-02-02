import re
import csv
import xml.etree.ElementTree as ET


def parse(txt: str) -> str:
    """Use regular expression to filter out HTTP header from request or response text"""

    regexp = re.compile(r"(.*\n)([\w-]+: .*)((?:\n.+)+)")
    try:
        return regexp.match(txt)[0]
    except Exception as e:
        print(f"{e}: ", txt)


def parseXML(filepath: str, outputDir: str):
    """Parses HTTP header data from Burp generated XML file"""

    mytree = ET.parse(filepath)
    root = mytree.getroot()
    totalItems = len(root)
    
    # Create CSV file with predifined headers
    f = open(f"{outputDir}/dataset_{totalItems}.csv", mode='w', newline='')
    csvWriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvWriter.writerow(['request', 'response'])

    # Iterate over items in XML, which should contain request/response pairs
    for i, child in enumerate(root.iter('item')):
        req, resp = None, None

        # Find request and response tags in item
        for node in child.iter():
            if node.tag == 'request' and node.text:
                req = parse(node.text)
            elif node.tag == 'response' and node.text:
                resp = parse(node.text)
            else:
                continue
        
        # Write only pairs 
        if req and resp:
            csvWriter.writerow([req, resp])
        print(f"Parsing {i+1}/{totalItems}", end = '\r')

    f.close()


if __name__ == "__main__":
    parseXML(filepath = "./data/http_burp_3.xml", outputDir="./data")
