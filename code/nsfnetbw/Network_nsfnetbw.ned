package netsimulator;
network  Network_nsfnetbw
{
    parameters:
        int numNodes = 14;
        int numTx = 14;
    types:
        channel Channel10kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 10kbps;
        }
        channel Channel40kbps extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 40kbps;
        }
    submodules:
        statistic: Statistic {
            numTx = numTx;
            numNodes = numNodes; 
        } 
        node0: Server {
            id = 0;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node1: Server {
            id = 1;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node2: Server {
            id = 2;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node3: Server {
            id = 3;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node4: Server {
            id = 4;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node5: Server {
            id = 5;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node6: Server {
            id = 6;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        node7: Server {
            id = 7;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node8: Server {
            id = 8;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node9: Server {
            id = 9;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node10: Server {
            id = 10;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[4];
        }
        node11: Server {
            id = 11;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node12: Server {
            id = 12;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[3];
        }
        node13: Server {
            id = 13;
            numTx = numTx;
            numNodes = numNodes;
            gates:
                port[2];
        }
        tController: NetTrafficController {
            numTx = numTx;
            numNodes = numNodes; 
            gates:
                out[numTx];
         }

    connections:

      node0.port[0] <--> Channel10kbps <--> node1.port[0];
      node0.port[1] <--> Channel10kbps <--> node3.port[0];
      node0.port[2] <--> Channel10kbps <--> node2.port[0];
      node1.port[1] <--> Channel10kbps <--> node2.port[1];
      node1.port[2] <--> Channel10kbps <--> node7.port[0];
      node2.port[2] <--> Channel10kbps <--> node5.port[0];
      node3.port[1] <--> Channel40kbps <--> node4.port[0];
      node3.port[2] <--> Channel10kbps <--> node8.port[0];
      node4.port[1] <--> Channel40kbps <--> node5.port[1];
      node4.port[2] <--> Channel10kbps <--> node6.port[0];
      node5.port[2] <--> Channel10kbps <--> node12.port[0];
      node5.port[3] <--> Channel10kbps <--> node13.port[0];
      node6.port[1] <--> Channel10kbps <--> node7.port[1];
      node7.port[2] <--> Channel40kbps <--> node10.port[0];
      node8.port[1] <--> Channel10kbps <--> node9.port[0];
      node8.port[2] <--> Channel10kbps <--> node11.port[0];
      node9.port[1] <--> Channel10kbps <--> node10.port[1];
      node9.port[2] <--> Channel10kbps <--> node12.port[1];
      node10.port[2] <--> Channel10kbps <--> node11.port[1];
      node10.port[3] <--> Channel10kbps <--> node13.port[1];
      node11.port[2] <--> Channel10kbps <--> node12.port[2];


      tController.out[0] --> { @display("ls=grey,1,d"); } --> node0.tControl;
      tController.out[1] --> { @display("ls=grey,1,d"); } --> node1.tControl;
      tController.out[2] --> { @display("ls=grey,1,d"); } --> node2.tControl;
      tController.out[3] --> { @display("ls=grey,1,d"); } --> node3.tControl;
      tController.out[4] --> { @display("ls=grey,1,d"); } --> node4.tControl;
      tController.out[5] --> { @display("ls=grey,1,d"); } --> node5.tControl;
      tController.out[6] --> { @display("ls=grey,1,d"); } --> node6.tControl;
      tController.out[7] --> { @display("ls=grey,1,d"); } --> node7.tControl;
      tController.out[8] --> { @display("ls=grey,1,d"); } --> node8.tControl;
      tController.out[9] --> { @display("ls=grey,1,d"); } --> node9.tControl;
      tController.out[10] --> { @display("ls=grey,1,d"); } --> node10.tControl;
      tController.out[11] --> { @display("ls=grey,1,d"); } --> node11.tControl;
      tController.out[12] --> { @display("ls=grey,1,d"); } --> node12.tControl;
      tController.out[13] --> { @display("ls=grey,1,d"); } --> node13.tControl;
}