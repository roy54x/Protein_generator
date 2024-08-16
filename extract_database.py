import os
import pandas as pd
import requests
from Bio import PDB
import numpy as np
from Bio.SeqUtils import seq1
from lxml import etree

pdb_list = PDB.PDBList()
parser = PDB.PDBParser()


def get_uniprot_dataframe(xml_file):
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

        try:
            pdb_ids = fetch_pdb_ids(accession)
            if pdb_ids:
                print(sequence)
                pdb_file = download_pdb(pdb_ids[0])
                structure = parser.get_structure(pdb_ids[0], pdb_file)  # Use the first PDB file
                coords = extract_amino_acid_coords(structure)
                contact_map = get_contact_map_from_coords(coords)
                structure_info = get_structure_info(structure)
            else:
                coords = None
                contact_map = None
                structure_info = None
        except Exception as e:
            print(f"Error processing {accession}: {e}")
            contact_map = None
            structure_info = None

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
            "coords": coords,
            "contact_map": contact_map,
            "structure_info": structure_info
        })
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return pd.DataFrame(data)


def fetch_pdb_ids(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {uniprot_id}")
        return None

    # Parse XML to find PDB IDs
    tree = etree.fromstring(response.content)
    pdb_ids = []
    for db_reference in tree.findall(".//{http://uniprot.org/uniprot}dbReference[@type='PDB']"):
        pdb_id = db_reference.get('id')
        pdb_ids.append(pdb_id)
    if pdb_ids:
        print(pdb_ids)
    return pdb_ids


def download_pdb(pdb_id, pdb_dir='pdb_files'):
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    pdb_list.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
    return os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')


def extract_amino_acid_coords(structure):
    ca_coords = []
    for model in structure:
        for chain in model:
            if chain.id != "A":
                continue
            for residue in chain:
                # Filter out non-amino acid residues
                if residue.id[0] == ' ' and 'CA' in residue:
                    ca_atom = residue['CA']
                    print(seq1(residue.resname))
                    ca_coords.append(ca_atom.get_coord())
    return np.array(ca_coords)


def get_contact_map_from_coords(ca_coords, threshold=8.0):
    num_residues = len(ca_coords)
    contact_map = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            distance = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if distance < threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1

    return contact_map

# 5. Function to extract structure info
def get_structure_info(structure):
    return {
        "num_chains": len(list(structure.get_chains())),
        "num_residues": len(list(structure.get_residues())),
        "num_atoms": len(list(structure.get_atoms()))
    }


df = get_uniprot_dataframe(r"D:\python project\data\uniprot_sprot.xml\uniprot_sprot.xml")
df.to_csv("protein_df.csv", index=False)

print(df.head())
