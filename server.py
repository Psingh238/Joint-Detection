#    Copyright 2023-2024 Seattle University Team ECE 24.2
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import socketserver

class DataHandler(socketserver.BaseRequestHandler):
    
    # Setup constant parameters used to figure out row endings and transmission endings and expected number of rows
    def setup(self):
        self.MARKER = b'\r\n'
        self.STOP_CODE = b'\0'
        self.NUM_ROWS = 22
        
    def handle(self):
        
        # define variables for handling parsing of data
        rows_received = 0
        stop_flag = 0
        data = b''
        full_pose = []
        
        # loops and handles continuous data from the client
        while not stop_flag:
            # loops until all rows are parsed for one set of data from one frame
            while rows_received < self.NUM_ROWS:
                # Read up to 1024 bytes from the client and add into a temporary variable for later use
                chunk = self.request.recv(4096)
                
                # break out of loop since nothing was received
                if not chunk:
                    break
                data += chunk
                
                # sets the stop flag to allow handler to exit without prematurely closing the connection
                if data.count(self.STOP_CODE):
                    stop_flag = 1

                # Process each row as it is received
                while self.MARKER in data:
                    
                    # split data into a row in binary string and store the rest into the same variable
                    # for further iteration
                    row, data = data.split(self.MARKER, 1)
                    
                    # if row was empty, then we continue looking for more rows
                    if not row:
                        continue
                    
                    # decode and split the row using the comma to separate all the values
                    parsed_data = row.decode().split(',')
                    
                    # convert from string to either integer or float depending on the type of data present
                    # also append it to the list full_pose
                    full_pose.append([int(num) if num.isdecimal() else float(num) for num in parsed_data])
                    
                    # update the rows_received variable to be able to signal the outer loop whether we need to receive
                    # another chunk of data or not
                    rows_received += 1
                    
                    # break out of the loop when all data is parsed and print the result for the frame to the console
                    if rows_received == self.NUM_ROWS:
                        print(full_pose)
                        break
            
            # reset all variables for next batch of data
            rows_received = 0
            full_pose = []
            
        print('Stop code detected, leaving handler')
            
        
        
 
# function to get WLAN IP to display to user so that they can connect to the server
def wlan_ip():
    import subprocess
    result=subprocess.run('ipconfig',stdout=subprocess.PIPE,text=True).stdout.lower()
    scan=0
    for i in result.split('\n'):
        if 'wireless' in i: scan=1
        if scan:
            if 'ipv4' in i: return i.split(':')[1].strip()

if __name__ == "__main__":
    # Run the server listening on all interfaces at port 5000
    HOST, PORT = '0.0.0.0', 5000

    print(f'Server running at {wlan_ip()}:{PORT}')
    print('Hit Ctrl + C to stop server.')

    # Create the server, binding to localhost on port 5000
    with socketserver.TCPServer((HOST, PORT), DataHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()