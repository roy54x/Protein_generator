import pandas as pd
from lxml import etree


def parse_uniprot_xml(xml_file):
    data = []
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')
    for event, elem in context:
        accession = elem.findtext('{http://uniprot.org/uniprot}accession')
        sequence = elem.findtext('{http://uniprot.org/uniprot}sequence')
        protein_names = [name.text for name in elem.findall('.//{http://uniprot.org/uniprot}fullName')]
        organism = elem.findtext('.//{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}name')
        function = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="function"]/{http://uniprot.org/uniprot}text')
        subcellular_location = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="subcellular location"]/{http://uniprot.org/uniprot}subcellularLocation/{http://uniprot.org/uniprot}location')
        tissue_specificity = elem.findtext('.//{http://uniprot.org/uniprot}comment[@type="tissue specificity"]/{http://uniprot.org/uniprot}text')
        domain_structure = [domain.get('description') for domain in
                            elem.findall('.//{http://uniprot.org/uniprot}feature[@type="domain"]')]
        ptms = [ptm.get('description') for ptm in
                elem.findall('.//{http://uniprot.org/uniprot}feature[@type="modified residue"]')]
        interactions = [interaction.text for interaction in
                        elem.findall('.//{http://uniprot.org/uniprot}interactant/{http://uniprot.org/uniprot}geneName')]
        sequence_annotations = [(annot.get('description'), annot.get('evidence')) for annot in
                                elem.findall('.//{http://uniprot.org/uniprot}feature')]

        data.append({
            'accession': accession,
            'sequence': sequence,
            'protein_names': protein_names,
            'organism': organism,
            'function': function,
            'subcellular_location': subcellular_location,
            'tissue_specificity': tissue_specificity,
            'domain_structure': domain_structure,
            'ptms': ptms,
            'interactions': interactions,
            'sequence_annotations': sequence_annotations,
        })
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return pd.DataFrame(data)


if __name__ == '__main__':
    xml_file_path = r'C:\Users\RoyIlani\Downloads\uniprot_sprot.xml'
    protein_df = parse_uniprot_xml(xml_file_path)
    protein_df.to_csv("protein_df.csv")
