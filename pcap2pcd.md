

    rostopic echo /velodyne_points |more

    roslaunch velodyne_pointcloud 32e_points.launch pcap:=/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/Raw-001-HDL32.pcap pcap_time:=true read_once:=tru

    rosrun pcl_ros pointcloud_to_pcd input:=/velodyne_points    

