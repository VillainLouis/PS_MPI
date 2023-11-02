import pickle

async def send_data(comm, data, dst_rank, tag_epoch):
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  
    comm.send(data, dest=dst_rank, tag=tag_epoch)
    # print("after send")

async def get_data(comm, src_rank, tag_epoch):
    data = comm.recv(source=src_rank, tag=tag_epoch)
    data = pickle.loads(data)
    return data
