from rdkit.Chem import MACCSkeys,rdFingerprintGenerator,AllChem,ChemicalFeatures,PandasTools,MolToSmiles
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit import RDConfig
import pandas as pd
import numpy as np
import os
from mordred import Calculator, descriptors
from json import load
from sklearn.metrics import pairwise_distances
class BaseFPGen():

    def __init__(self):
        self.fp_size = 2048
        self.fp_name = ""
    def generate_fp(self,mol):
        return self.fpgen.GetFingerprint(mol)
    def create_col_names(self):
        return [f"{self.fp_name}_bit_{i}"for i in range(self.fp_size)]
    def create_fp_dataframe(self,mols):
        fp_lists =  [ self.generate_fp(mol).ToList() for mol in mols]
        return pd.DataFrame(fp_lists,columns=self.create_col_names())
class PharmaCore2DGPGen(BaseFPGen):
    def __init__(self):
        super().__init__()
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')

        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        self.fpgen = SigFactory(factory,minPointCount=2,maxPointCount=3,trianglePruneBins=False)

        self.fpgen.SetBins([(0,2),(2,5),(5,8)])
        self.fpgen.Init()
        self.fp_size = self.fpgen.GetSigSize()
        self.fp_name = "2Dpharmacore"
    def generate_fp(self,mol):
        return Generate.Gen2DFingerprint(mol,self.fpgen)
class RDkitFPGen(BaseFPGen):
    def __init__(self):
        self.fp_size = 2048
        self.fp_name = "RDKitFP"
        self.fpgen = rdFingerprintGenerator.GetRDKitFPGenerator()
class MorganFPGen(BaseFPGen):
    def __init__(self):
        self.fp_size = 4096
        self.fp_name = "MorganFP"
        self.fpgen = AllChem.GetMorganGenerator(radius=5,fpSize=self.fp_size)
class MACCSFPGen(BaseFPGen):
    def __init__(self):
        self.fp_size = 167
        self.fp_name = "MACCSFP"
    def generate_fp(self,mol):
        return MACCSkeys.GenMACCSKeys(mol)
def create_descriptors(mol,descriptors):
    cal = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(mol)
    return df
def create_fp_features(mols,fps_to_generate=None):
    """ 
        takes in a iterable of rdkit molecules and creates a dataframe containing its fingerprints,
        then converts them to avg pw distances
    """
    pw_fp_distances = {}
    if fps_to_generate is None:
        fps_to_generate = [PharmaCore2DGPGen(),RDkitFPGen(),MorganFPGen(),MACCSFPGen()]
    for gen in fps_to_generate:
        fp_dataframe = gen.create_fp_dataframe(mols)
        pw_fp_distances[gen.fp_name] =  calculate_pairwise_fp_distances(fp_dataframe).flatten()
        
    return pd.DataFrame(pw_fp_distances)
def calculate_pairwise_fp_distances(fp_df):
    pw_dist = pairwise_distances(fp_df, n_jobs = -1)
    avg_pw_dist = np.mean(pw_dist, axis=0).reshape(-1, 1)
    return avg_pw_dist
def load_and_preprocess(filename,columns_to_remove_path):
    """ 
        load an sdf file, annotate by provided compound_id and create 
        molecular descriptors/fingerprint

    """
    dfmols = PandasTools.LoadSDF(filename)
    fp_distances = create_fp_features(dfmols["ROMol"])
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(list(dfmols["ROMol"]))
    df.set_index(dfmols["compound_id"])
    columns_to_remove = load(open(columns_to_remove_path,"rb"))
    df = df.drop(columns_to_remove,axis=1)
    return pd.concat([df,fp_distances],axis=1) 
def remove_mol_duplicates(df):
    df["smiles"] = [MolToSmiles(mol) for mol in df["ROMol"]]
    duplicates = df["smiles"].duplicated(keep=False)
    df =df[~duplicates]
    df.drop(["smiles"],axis=1)
    return df
def load_and_preprocess_train_data_set(filename,columns_to_remove_path):
    """ 
        load an sdf file, annotate by provided compound_id and create 
        molecular descriptors/fingerprint

    """
    dfmols = PandasTools.LoadSDF(filename)
    dfmols = remove_mol_duplicates(dfmols)
    fp_distances = create_fp_features(dfmols["ROMol"])
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(list(dfmols["ROMol"]))
    df.set_index(dfmols["compound_id"])
    columns_to_remove = load(open(columns_to_remove_path,"rb"))
    df = df.drop(columns_to_remove,axis=1)
    return pd.concat([df,fp_distances],axis=1),dfmols["pLC50"].astype("float")
