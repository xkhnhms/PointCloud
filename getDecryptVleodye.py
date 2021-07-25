import struct
import numpy as np

def shift(value ,key):
    return ((value+key)&0xFF)

def decrypt(value,key):
    return shift(value,-key)

data_file='./data/Raw-001-HDL32.pcap'
xxx_file='./data/raw-1-HDL32.pcap'

# xxx_file='/media/fs/Elements SE/lidar_pcap/raw-1-HDL32.pcap'

xxx = open(xxx_file,'wb')
data = open(data_file,'rb')

h = data.read(24)
xxx.write(h)

print('-----header----')
print(h)
print(int.from_bytes(h, 'little'))

count=0

while data is not None:
    time_s = data.read(8)

    time_s_len=int.from_bytes(time_s, 'little')

    print(type(time_s_len),time_s_len)

    if time_s_len < 10:
        break

    xxx.write(time_s)
    len_sec = data.read(4)  # caplen
    xxx.write(len_sec)

    # print(int.from_bytes(len_sec, 'little'))

    len_sec2 = data.read(4)
    xxx.write(len_sec2)

    print(int.from_bytes(len_sec2, 'little'))

    len_data = int.from_bytes(len_sec2, 'little')
    len_sec3 = data.read(len_data)

    if len_data!=1248:
        # continue
        xxx.write(len_sec3)

    else:

        xxx.write(len_sec3[:len_data-1206])

        # print('----len_sec3------')
        # print(len(len_sec3))

        count+=1
        print(count,int.from_bytes(len_sec, 'little'),len_data)

        btv = []
        # btv = ''
        for i in range(len_data-1206,len_data):
            tv = decrypt(len_sec3[i],1)
            # btv+=str(tv)
            btv.append(tv)
            # print('-------'+str(i)+'--------')
        # print(btv.encode())

        # print(len(btv))
        # xxx.write(bytes(btv.encode()))
        xxx.write(bytes(btv))

print('Done!')

# while data:
#     time_s = data.read(8)
#     len_sec = data.read(4)          # caplen
#
#     print(int.from_bytes(len_sec,'little'))
#
#     len_sec2 = data.read(4)
#     print(int.from_bytes(len_sec2,'little'))
#
#     len_data = int.from_bytes(len_sec2,'little')
#     len_sec3 = data.read(len_data)
#     print(int.from_bytes(len_sec3,'little'))
#
#     for i in range(len_data[:-1206]):
#         tv = decrypt(len_data[i], 1)
#         print(tv)

