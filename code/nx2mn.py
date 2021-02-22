from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
import networkx as nx
import random
import os
import tarfile,traceback
import numpy as np

path = 'nsfnetbw/graph_attr.txt'
G = nx.Graph()
cat_mat = []

class DataNet:
    def _generate_graphs_dic(self, path,file):
        global G
        graphs_dic = {}
        G = nx.read_gml(path + file, destringizer=int)
        graphs_dic[file] = G
            
        return graphs_dic

    def readData(self, data_dir):
        global G
        tuple_files = []
        file_dict = {}
        for root, dirs, files in os.walk(data_dir):
            files.sort()
            tuple_files.extend([(root, f) for f in files if f.endswith("tar.gz")])
            print(tuple_files)
            it = 0
            for root, file in tuple_files:
                try:
                    if it == 1:
                        return                   
                    tar = tarfile.open(os.path.join(root,file), 'r:gz')
                    dir_info = tar.next()
                    input_files = tar.extractfile(dir_info.name + '/input_files.txt')
                    ran_num = random.randint(0,100)
                    file = input_files.readlines()[ran_num].split(';')
                    file_dict[ran_num] = (file[1],file[2])
                    graph_file_path = root+"/graphs/"
                    graph_file = file_dict[ran_num][0]
                    self._generate_graphs_dic(graph_file_path, graph_file)
                except:
                    traceback.print_exc()
                    print ("Error in the file:" +file)
                    print ("iteration: " +str(it))
                    exit()
                it+=1

    def create_Cap_Mat(self):
        global cap_mat
        cap_mat = np.full((G.number_of_nodes()+1, G.number_of_nodes()+1), fill_value=None)
        for node in range(G.number_of_nodes()):
            for adj in G[node]:
                cap_mat[node, adj] = G[node][adj][0]['bandwidth']
        # info(cap_mat)

def MyNetwork():
    global G
    net = Mininet( topo=None,
                   build=False,
                   ipBase='10.0.0.0/8')

    info( '*** Adding controller\n' )
    c0=net.addController(name='c0',
                      controller=RemoteController,
                      protocol='tcp',
                      port=6633,ip='127.0.0.1')

    # print('G nodes = ', G.nodes())
    for n in G.nodes():
        net.addSwitch("s%s" % str(n+1))
        if int(n) in list(G.nodes()):
            net.addHost('h%s' % str(n+1))
            net.addLink('s%s' % str(n+1), 'h%s' % str(n+1))
    # print('edges = ', G.edges())
    for (n1,n2) in G.edges():
        net.addLink('s%s' % str(n1+1), 's%s' % str(n2+1),cls=TCLink)

    info( '*** Starting network\n')
    net.build()
    info( '*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info( '*** Starting switches\n')
    for n in range(0,G.number_of_nodes()):
        net.get('s%s' % str(n+1)).start([c0])

    info( '*** Post configure switches and hosts\n')

    CLI(net)
    net.stop()

setLogLevel('info')
datanet = DataNet()
datanet.readData(path)
datanet.create_Cap_Mat()

MyNetwork()


    
