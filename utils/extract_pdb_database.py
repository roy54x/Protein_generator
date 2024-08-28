import itertools
import json
import os
import pandas as pd
import requests
from Bio import PDB
import numpy as np
from Bio.SeqUtils import seq1
from lxml import etree

from utils.constants import AMINO_ACIDS, MAIN_DIR

pdb_list = PDB.PDBList()
parser = PDB.PDBParser()


def get_pdb_ids_from_uniprot_xml(xml_file, json_path='pdn_ids.json'):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    accessions = data.keys()
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')

    counter = 0
    for event, elem in context:
        accession = elem.findtext('{http://uniprot.org/uniprot}accession')
        print("processing protein: " + accession)
        if accession in accessions:
            continue

        data[accession] = fetch_pdb_ids(accession)

        # Write to JSON every 100 iterations
        if (counter + 1) % 100 == 0:
            with open(json_path, 'w') as f:
                json.dump(data, f)
        counter += 1

    # Write any remaining data to JSON
    with open(json_path, 'w') as f:
        json.dump(data, f)

    return list(set(itertools.chain(*data.values())))  # Remove duplicates


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


def get_pdb_data(pdb_ids, output_path, dataframe_dir_name="pdb_data", num_samples_in_df=50000):
    data = []
    file_index = 0

    dataframe_output_dir = os.path.join(output_path, dataframe_dir_name)
    os.makedirs(dataframe_output_dir, exist_ok=True)

    for pdb_id in pdb_ids:
        try:
            pdb_file = download_pdb(pdb_id, os.path.join(output_path, 'pdb_files'))
            structure = parser.get_structure(pdb_id, pdb_file)
            structure_info = get_structure_info(structure)
            chain_ids, chain_sequences, chain_coords = extract_amino_acid_chains(structure)

            for chain_id, sequence, coords in zip(chain_ids, chain_sequences, chain_coords):
                data.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'sequence': sequence,
                    "coords": coords,
                    "structure_info": structure_info
                })

            if len(data) >= num_samples_in_df:
                df = pd.DataFrame(data)
                save_dataframe(df, dataframe_output_dir, file_index)
                file_index += 1
                data = []  # Reset data

        except Exception as err:
            print(err)

    # Save remaining data if any
    if data:
        df = pd.DataFrame(data)
        save_dataframe(df, dataframe_output_dir, file_index)


def save_dataframe(df, output_path, file_index):
    df.to_csv(os.path.join(output_path, f"pdb_df_{file_index}.csv"), index=False)
    df.to_json(os.path.join(output_path, f"pdb_df_{file_index}.json"), orient='records', lines=True)


def download_pdb(pdb_id, pdb_dir='pdb_files'):
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    pdb_file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
    if not os.path.exists(pdb_file_path):
        pdb_list.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
    return pdb_file_path


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


# 5. Function to extract structure info
def get_structure_info(structure):
    return {
        "num_chains": len(list(structure.get_chains())),
        "num_residues": len(list(structure.get_residues())),
        "num_atoms": len(list(structure.get_atoms()))
    }


pdb_ids = get_pdb_ids_from_uniprot_xml(os.path.join(MAIN_DIR, r"UniProt\uniprot_sprot.xml\uniprot_sprot.xml"),
                                       os.path.join(MAIN_DIR, "PDB", "UniProt2PBD.json"))
get_pdb_data(pdb_ids, output_path=os.path.join(MAIN_DIR, "PDB"))
