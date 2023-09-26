import os
import os.path as osp
import glob
import numpy as np
from pymatreader import read_mat
import tqdm
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from sklearn.model_selection import train_test_split
import h5py
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as img
#np.random.seed(1337)
from label_text import text

root_path = './'
stat_path = osp.join(root_path, 'statistics')
subjectid_file = osp.join(stat_path, 'subjectid.txt')
subjectname_file = osp.join(stat_path, 'subjectname.txt')
label_file = osp.join(stat_path, 'label.txt')

def load_skes():
    skes_mat_list = glob.glob("/home/user/soojie/SGN_/data/sysu/SYSU3DAction/3DvideoNorm/*/*/sklWorld.mat")
    
    skes_all = []
    file_names = []
    frame_cnt=[]
    actions=[]
    names=[]
    for sample_i, skes_mat_name in enumerate(skes_mat_list):
        file_names.append("/".join(skes_mat_name.split("/")[-3:-1]))
        name=skes_mat_name.split("/")[-3:-1][0]
        action=skes_mat_name.split("/")[-3:-1][1].split('video')[-1]
        actions.append(action)
        names.append(name)
        mat_data = read_mat(skes_mat_name)
        skes = mat_data['SW']
        skes = skes.transpose(2, 0, 1)
        skes = skes - skes[:, 1:2, :] #[frame,20,3]
        skes = skes.reshape(-1,20*3) #[frame,20*3]
        skes_all.append(skes)
        frame_cnt.append(skes.shape[0])
        #with open("frame_cnt.txt", "a", encoding="utf-8") as f:
        #    f.write(str(skes.shape[0])+'\n')
    np.savez("sysu.npz", 
                joints=np.array(skes_all, dtype=object),
                names=names,
                actions=actions
            )
   
    return skes_all, frame_cnt
            
def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        #print(ske_joints.shape)#[frame,20*3]
        num_frames = ske_joints.shape[0]
        
        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :60] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            ske_joints[f] -= np.tile(origin, 20)

        skes_joints[idx] = ske_joints  # Update
    return skes_joints


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max() #638
    print('max frame:',max_num_frames)
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 20*3), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        aligned_skes_joints[idx, :num_frames] = ske_joints
        
    return aligned_skes_joints

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 12))
    for idx, l in enumerate(labels):  
        labels_vector[idx, l] = 1

    return labels_vector

def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices
    
def get_indices(performer, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        values = list(range(1, 41))
        train_ids = [x for x in values if x % 2 == 0]
        test_ids = [x for x in values if x % 2 != 0]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int)
            
    else:  # Same Subject IDs
        values = list(range(1, 41))

        for val in values:
            idx = np.where(performer == val)[0]
            half = int(len(idx)/2)
            np.hstack((train_indices, idx[:half])).astype(np.int)
            np.hstack((test_indices, idx[half:])).astype(np.int)
       
    return train_indices, test_indices
    
def split_dataset(skes_joints, evaluation):
    performer = np.loadtxt(subjectid_file, dtype=np.int)
    
    sysu_data = np.load('sysu.npz',allow_pickle=True)
    joints = sysu_data['joints']
    names = sysu_data['names']
    actions = sysu_data['actions']
    
    train_indices, test_indices = get_indices(performer, evaluation)
    m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    train_indices, val_indices = split_train_val(train_indices, m)
    # Save labels and num_frames for each sequence of each data set
    
    train_labels = actions[train_indices].astype(int)-1
    val_labels = actions[val_indices].astype(int)-1
    test_labels = actions[test_indices].astype(int)-1
    
    print(len(train_labels)) #228
    print(len(val_labels)) #12
    print(len(test_labels)) #240
    #test_labels=[]
    #for i in range(0,joints.shape[0]):
    #    idx=int(actions[i])-1
    #    test_labels.append(idx)
  
    # Save data into a .h5 file
    h5file = h5py.File(osp.join('SYSU_%s.h5' % (evaluation)), 'w')
     # Training set
    h5file.create_dataset('x', data=skes_joints[train_indices])
    train_one_hot_labels = one_hot_vector(train_labels)
    h5file.create_dataset('y', data=train_one_hot_labels)
    # Validation set
    h5file.create_dataset('valid_x', data=skes_joints[val_indices])
    val_one_hot_labels = one_hot_vector(val_labels)
    h5file.create_dataset('valid_y', data=val_one_hot_labels)
    # Test set
    h5file.create_dataset('test_x', data=skes_joints[test_indices])
    test_one_hot_labels = one_hot_vector(test_labels)
    h5file.create_dataset('test_y', data=test_one_hot_labels)

    h5file.close()


def visualize(skes_joints):
    #skes_joints=skes_joints.reshape(480,-1)
    n_components = 2
    model = TSNE(n_components=n_components)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink', 'lime', 'brown']
    skes=np.load('sysu.npz',allow_pickle=True)
    
    for i in range(0,skes['joints'].shape[0]):
        joints=skes['joints'][i]
        fit=model.fit_transform(joints.data)
        if int(skes['actions'][i])-1 < 3:
            plt.scatter(fit[:, 0], fit[:, 1],c=colors[int(skes['actions'][i])-1])
        print(i)
    
    plt.savefig('scatter.jpg')
    plt.close()
    
    num_colors = len(colors)
    image_width = 40
    image_height = 30
    fig, ax = plt.subplots(figsize=(image_width / 10, image_height / 10))
    for i, color in enumerate(colors):
        x = i % 10
        y = i // 10
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
        ax.text(x + 0.5, y + 0.5, str(i), ha='center', va='center', fontsize=8, color='black')


    ax.set_xlim(0, 10)
    ax.set_ylim(0, num_colors // 10 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('c.jpg')
    plt.close()
            
            
if __name__ == "__main__":
    skes_joints,frames_cnt = load_skes()
    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints, np.array(frames_cnt))
    #visualize(skes_joints)
    evaluations = ['CS', 'SS']
    for evaluation in evaluations:
        split_dataset(skes_joints,evaluation)