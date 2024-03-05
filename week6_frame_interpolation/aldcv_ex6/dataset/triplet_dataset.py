import os
from PIL import Image
from torch.utils.data.dataset import Dataset


class TripletDataset(Dataset):
    """
    Each triplet has its folder with the corresponding 3 images.
    Structure of the triplet dataset:
        triplet_<number>
            -   1.png
            -   2.png
            -   3.png
    """
    def __init__(self, triplets_path, transform):
        self.triplets_path = triplets_path
        self.transform = transform
        self.triplets = list(os.listdir(triplets_path))
    
        

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        current_path = self.triplets[index] # e.g. triplet triplet_000010
        
        # Make sure you understand the __init__ function and the structure of the data first.

        # TASK 1: Read the triplet and make sure you use the self.transform
        frames = sorted(list(os.listdir(self.triplets_path+current_path)))
        frames = [self.transform(Image.open(self.triplets_path+current_path+"/"+x)) for x in frames]
        frame_1 = frames[0]
        frame_2 = frames[1]
        frame_3 = frames[2]
        return frame_1, frame_2, frame_3


if __name__=="__main__":
    triplet_path = os.path.join(os.path.dirname(__file__), '../data/DAVIS_Triplets/train/')
    ds = TripletDataset(triplets_path=triplet_path,transform=None)
    F1,F2,F3 = ds.__getitem__(100)
    print(F1,F2,F3)

    
