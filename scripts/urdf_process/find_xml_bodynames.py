import xml.etree.ElementTree as ET

def extract_body_names(file_path):
    """
    Extracts all body names from an XML file.
    
    Parameters:
        file_path (str): The path to the XML file.

    Returns:
        list: A list of body names.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # List to store body names
    body_names = []
    
    # Iterate over all 'body' elements in the XML
    for body in root.findall(".//body"):
        name = body.get('name')
        if name:
            body_names.append(name)
    
    return body_names

# Example usage
xml_file_path = "/media/tairanh/lecar4tb/tairanh/Workspace/RoboVerse/roboverse/data/robots/ea/atlas_r1_nub_hand.xml"  # Replace with your XML file path

# Extract body names
body_names = extract_body_names(xml_file_path)

# Print the results
print("Body Names:")
for name in body_names:
    print(name)

print("bdo_names:", body_names)
print("num_bodies:", len(body_names))
