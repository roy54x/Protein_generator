import os
import pandas as pd
import requests
from Bio import PDB
import numpy as np
from Bio.SeqUtils import seq1
from lxml import etree

pdb_list = PDB.PDBList()
parser = PDB.PDBParser()

amino_acid_letters = [
    "A",  # Alanine
    "R",  # Arginine
    "N",  # Asparagine
    "D",  # Aspartic Acid
    "C",  # Cysteine
    "E",  # Glutamic Acid
    "Q",  # Glutamine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "L",  # Leucine
    "K",  # Lysine
    "M",  # Methionine
    "F",  # Phenylalanine
    "P",  # Proline
    "S",  # Serine
    "T",  # Threonine
    "W",  # Tryptophan
    "Y",  # Tyrosine
    "V"   # Valine
]

def get_pdb_ids_from_uniprot_xml(xml_file):
    pdb_ids = []
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')
    for event, elem in context:
        accession = elem.findtext('{http://uniprot.org/uniprot}accession')
        pdb_ids.extend(fetch_pdb_ids(accession))
        if len(pdb_ids) > 10:
            break
    return list(set(pdb_ids))  # Remove duplicates


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
    return pdb_ids


def get_pdb_data(pdb_ids):
    data = []
    for pdb_id in pdb_ids:
        pdb_file = download_pdb(pdb_id)
        structure = parser.get_structure(pdb_id, pdb_file)  # Use the first PDB file
        sequence, coords = extract_amino_acid_coords(structure)
        contact_map = get_contact_map_from_coords(coords)
        structure_info = get_structure_info(structure)

        data.append({
            'pdb_id': pdb_id,
            'sequence': sequence,
            "coords": coords,
            "contact_map": contact_map,
            "structure_info": structure_info
        })
    return pd.DataFrame(data)


def download_pdb(pdb_id, pdb_dir='pdb_files'):
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    pdb_list.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
    return os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')


def extract_amino_acid_coords(structure):
    sequence = []
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Filter out non-amino acid residues
                letter = seq1(residue.resname)
                if residue.id[0] == ' ' and 'CA' in residue and letter in amino_acid_letters:
                    sequence.append(letter)
                    ca_atom = residue['CA']
                    ca_coords.append(ca_atom.get_coord())
            break
    return ''.join(sequence), np.array(ca_coords)


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


pdb_ids = get_pdb_ids_from_uniprot_xml(r"D:\python project\data\uniprot_sprot.xml\uniprot_sprot.xml")
df = get_pdb_data(pdb_ids)
df.to_csv("protein_df.csv", index=False)
print(df.head())
