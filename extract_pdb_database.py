import os
import pandas as pd
import requests
from Bio import PDB
import numpy as np
from Bio.SeqUtils import seq1
from lxml import etree

from constants import AMINO_ACIDS

pdb_list = PDB.PDBList()
parser = PDB.PDBParser()


def get_pdb_ids_from_uniprot_xml(xml_file):
    pdb_ids = []
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')
    for event, elem in context:
        accession = elem.findtext('{http://uniprot.org/uniprot}accession')
        pdb_ids.extend(fetch_pdb_ids(accession))
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


def get_pdb_data(pdb_ids, output_path):
    data = []
    for pdb_id in pdb_ids:
        pdb_file = download_pdb(pdb_id, os.path.join(output_path, 'pdb_files'))
        structure = parser.get_structure(pdb_id, pdb_file)
        structure_info = get_structure_info(structure)
        chain_ids, chain_sequences, chain_coords = extract_amino_acid_chains(structure)

        for chain_id, sequence, coords in zip(chain_ids, chain_sequences, chain_coords):
            contact_map = get_contact_map_from_coords(coords)
            data.append({
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'sequence': sequence,
                "coords": coords,
                "contact_map": contact_map,
                "structure_info": structure_info
            })
    df = pd.DataFrame(data)
    df.to_json(os.path.join(output_path, "protein_df.json"), index=False)
    df.to_csv(os.path.join(output_path, "protein_df.csv"), index=False)
    return df


def download_pdb(pdb_id, pdb_dir='pdb_files'):
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    pdb_list.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
    return os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')


def extract_amino_acid_chains(structure):
    chain_ids = []
    chain_sequences = []
    chain_coords = []
    for model in structure:
        for chain in model:
            sequence = []
            coords = []
            for residue in chain:
                letter = seq1(residue.resname)
                # Filter out non-amino acid residues
                if residue.id[0] == ' ' and 'CA' in residue and letter in AMINO_ACIDS:
                    sequence.append(letter)
                    ca_atom = residue['CA']
                    coords.append(ca_atom.get_coord())
            chain_ids.append(chain.id)
            chain_sequences.append(''.join(sequence))
            chain_coords.append(np.array(coords))
    return chain_ids, chain_sequences, chain_coords


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
df = get_pdb_data(pdb_ids, output_path=r"D:\python project\data\PDB")
print(df.head())
