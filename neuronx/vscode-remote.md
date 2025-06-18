
Access through VS Code remote server
With Visual Studio Code installed on your local machine, you can use the Remote-SSH command to edit and run files that are stored on a Neuron instance. See the VS Code article for additional details.

Select Remote-SSH: Connect to Host… from the Command Palette (F1, ⇧⌘P)  
Enter in the full connection string from the ssh section above: ssh -i “/path/to/sshkey.pem” ubuntu@instance_ip_address  
VS Code should connect and automatically set up the VS Code server.  
Eventually, you should be prompted for a base directory. You can browse to a directory on the Neuron instance.  
In case you find that some commands seem greyed out in the menus, but the keyboard commands still work (⌘S to save or ^⇧` for terminal), you may need to restart VS Code.


![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-1.png)

![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-2.png)

![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-3.png)

![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-4.png)

![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-5.png)

![](https://github.com/gnosia93/xla-torch/blob/main/neuronx/images/vscode-remote-6.png)

