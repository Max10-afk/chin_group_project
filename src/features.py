from rdkit.Chem import MACCSkeys,rdFingerprintGenerator,AllChem,ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit import RDConfig
import pandas as pd
import os
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

def create_features(mols,fps_to_generate=None):
    """ 
        takes in a iterable of rdkit molecules and creates a dataframe containing its descriptors, fingerprints etc
    """
    fp_dfs = []
    if fps_to_generate is None:
        fps_to_generate = [PharmaCore2DGPGen(),RDkitFPGen(),MorganFPGen(),MACCSFPGen()]
    for gen in fps_to_generate:
        fp_dfs.append(gen.create_fp_dataframe(mols))
        features = pd.concat(fp_dfs,axis=1)
    return features
def load_and_preprocess(filename):
    pass