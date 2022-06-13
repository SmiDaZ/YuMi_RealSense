from yumipy import YuMiRobot
from SocketServer import *
host = 'localhost'
port = 777
addr = (host,port)

Last_Left = [""]
Last_Right = [""]

class MyTCPHandler(StreamRequestHandler):
    
    
    def handle(self):
        global Last_Left
        global Last_Right     
        self.data = self.request.recv(1024)
        if str(self.data) != "":
            self.c = str(self.data).split(" ")
            
            if self.c[6] == "Left":
                if self.c[7] == "Open":
                    if Last_Left == [""] or Last_Left == ["Close"]:
                        Last_Left = ["Open"]
                        print("Left Open Gripper")
                        y.left.open_gripper(wait_for_res=False)
                elif self.c[7] == "Close":
                    if Last_Left == [""] or Last_Left == ["Open"]:
                        Last_Left = ["Close"]
                        print("Left Close Gripper")
                        y.left.close_gripper(wait_for_res=False)
            if self.c[6] == "Right":
                if self.c[7] == "Open":
                    if Last_Right == [""] or Last_Right == ["Close"]:
                        Last_Right = ["Open"]
                        print("Right Open Gripper")
                        y.right.open_gripper(wait_for_res=False)
                elif self.c[7] == "Close":
                    if Last_Right == [""] or Last_Right == ["Open"]:
                        Last_Right = ["Close"]
                        print("Right Close Gripper")
                        y.right.close_gripper(wait_for_res=False)
                elif self.c[7] == "Ok":
                    print("Reset Home")
                    y.reset_home()
            y.right.goto_pose_delta((float(self.c[0]), float(self.c[1]), float(self.c[2])))
            y.left.goto_pose_delta((float(self.c[3]), float(self.c[4]), float(self.c[5])))
            

if __name__ == "__main__":
    y = YuMiRobot()
    server = TCPServer(addr, MyTCPHandler)
    
    print('starting server... for exit press Ctrl+C')

    server.serve_forever()
