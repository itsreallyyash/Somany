import streamlit as st

# Set page configuration for a blog layout and dark mode
st.set_page_config(
    page_title="OpenVPN Setup Guide",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark mode
st.markdown("""
    <style>
    body {
        color: #fff;
        background-color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #444;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fff;
    }
    p, ul, li, code, pre {
        color: #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

# Blog title and description
st.title("OpenVPN Setup Guide: A Step-by-Step Guide for macOS, Windows, and Linux")
st.write("""
This guide walks you through the process of setting up **OpenVPN** using **Ansible**.
We cover the steps for **macOS**, **Windows**, and **Linux**, with detailed instructions for each platform.
Whether you're setting up OpenVPN on an AWS EC2 instance or your own server, this guide will help you automate the process.
""")

# Step-by-Step Sections
st.header("1. Prerequisites")
st.write("""
Before you begin, ensure you have the following:
- **An Ubuntu Server**: Can be a local or cloud-based server (AWS, DigitalOcean, etc.)
- **SSH Access**: You'll need SSH access to your server (public IP/domain and a private SSH key).
- **Private Key File**: Typically a `.pem` file for AWS or a `.ppk` file for Windows.
- Basic familiarity with the terminal or command prompt.
""")

st.header("2. Install Ansible")

st.subheader("macOS")
st.code("""
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ansible
brew install ansible

# Verify installation
ansible --version
""", language='bash')

st.subheader("Windows")
st.code("""
# Install WSL (Windows Subsystem for Linux)
# Follow the steps in the official Microsoft docs: https://docs.microsoft.com/en-us/windows/wsl/install

# Install Ubuntu from Microsoft Store, then open Ubuntu terminal:
sudo apt update
sudo apt install ansible -y

# Verify installation
ansible --version
""", language='bash')

st.subheader("Linux (Ubuntu)")
st.code("""
# Update your system:
sudo apt update

# Install Ansible:
sudo apt install ansible -y

# Verify installation
ansible --version
""", language='bash')

st.header("3. Create the Ansible Playbook")
st.write("""
The playbook is the automation script that sets up OpenVPN. Follow these steps:
""")

st.code("""
# Create the playbook file
touch openvpn_setup.yml

# Add the following content:
---
---
- name: OpenVPN Setup Playbook
  hosts: vpn
  become: yes
  vars:
    openvpn_port: 1194
    openvpn_protocol: udp
    openvpn_server_ip: "10.8.0.0"
    openvpn_subnet_mask: "255.255.255.0"
    openvpn_dns1: "8.8.8.8"
    openvpn_dns2: "8.8.4.4"
    easy_rsa_dir: "/home/ubuntu/openvpn-ca"
    country: "US"
    province: "California"
    city: "San Francisco"
    org: "My Company"
    email: "admin@example.com"
    ou: "IT"
    common_name_ca: "yash1"
    common_name_server: "server"

  tasks:
    - name: Update and upgrade the system
      apt:
        update_cache: yes
        upgrade: dist
      become: yes

    - name: Install OpenVPN and Easy-RSA
      apt:
        name:
          - openvpn
          - easy-rsa
        state: present
      become: yes

    - name: Remove existing Easy-RSA directory if it exists
      file:
        path: "{{ easy_rsa_dir }}"
        state: absent
      become: yes
      ignore_errors: yes

    - name: Create Easy-RSA directory using make-cadir
      command: "make-cadir {{ easy_rsa_dir }}"
      become: yes
      args:
        creates: "{{ easy_rsa_dir }}"

    - name: Fix permissions on Easy-RSA directory
      file:
        path: "{{ easy_rsa_dir }}"
        owner: ubuntu
        group: ubuntu
        recurse: yes
      become: yes

    - name: Update vars file in Easy-RSA
      lineinfile:
        path: "{{ easy_rsa_dir }}/vars"
        regexp: "{{ item.regexp }}"
        line: "{{ item.line }}"
      loop:
        - {
            regexp: "^set_var EASYRSA_REQ_COUNTRY",
            line: 'set_var EASYRSA_REQ_COUNTRY "{{ country }}"',
          }
        - {
            regexp: "^set_var EASYRSA_REQ_PROVINCE",
            line: 'set_var EASYRSA_REQ_PROVINCE "{{ province }}"',
          }
        - {
            regexp: "^set_var EASYRSA_REQ_CITY",
            line: 'set_var EASYRSA_REQ_CITY "{{ city }}"',
          }
        - {
            regexp: "^set_var EASYRSA_REQ_ORG",
            line: 'set_var EASYRSA_REQ_ORG "{{ org }}"',
          }
        - {
            regexp: "^set_var EASYRSA_REQ_EMAIL",
            line: 'set_var EASYRSA_REQ_EMAIL "{{ email }}"',
          }
        - {
            regexp: "^set_var EASYRSA_REQ_OU",
            line: 'set_var EASYRSA_REQ_OU "{{ ou }}"',
          }
        - {
            regexp: "^set_var EASYRSA_KEY_SIZE",
            line: "set_var EASYRSA_KEY_SIZE 2048",
          }
        - {
            regexp: "^set_var EASYRSA_DIGEST",
            line: 'set_var EASYRSA_DIGEST "sha256"',
          }
      become: yes

    - name: Initialize PKI
      command: "./easyrsa init-pki"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      environment:
        EASYRSA_BATCH: "1"
      register: init_pki
      changed_when: "'init-pki' in init_pki.stdout"

    - name: Build Certificate Authority
      command: "./easyrsa build-ca nopass"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      environment:
        EASYRSA_BATCH: "1"
        EASYRSA_REQ_CN: "{{ common_name_ca }}"
      register: build_ca
      changed_when: "'CA creation complete' in build_ca.stdout"

    - name: Generate server certificate and key
      command: "./easyrsa gen-req server nopass"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      environment:
        EASYRSA_BATCH: "1"
        EASYRSA_REQ_CN: "{{ common_name_server }}"
      register: gen_req
      changed_when: "'Certificate request created' in gen_req.stdout"

    - name: Sign server certificate
      command: "./easyrsa sign-req server server"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      environment:
        EASYRSA_BATCH: "1"
      register: sign_cert
      changed_when: "'Signature ok' in sign_cert.stdout"

    - name: Generate Diffie-Hellman parameters
      command: "./easyrsa gen-dh"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      environment:
        EASYRSA_BATCH: "1"
      register: gen_dh
      changed_when: "'DH parameters generated' in gen_dh.stdout"

    - name: Generate HMAC key
      command: "openvpn --genkey --secret ta.key"
      args:
        chdir: "{{ easy_rsa_dir }}"
      become: yes
      register: gen_hmac
      changed_when: "'ta.key' in gen_hmac.stdout"

    - name: Copy CA certificate to OpenVPN directory
      copy:
        src: "{{ easy_rsa_dir }}/pki/ca.crt"
        dest: "/etc/openvpn/ca.crt"
        remote_src: yes
      become: yes

    - name: Copy server certificate to OpenVPN directory
      copy:
        src: "{{ easy_rsa_dir }}/pki/issued/server.crt"
        dest: "/etc/openvpn/server.crt"
        remote_src: yes
      become: yes

    - name: Copy server key to OpenVPN directory
      copy:
        src: "{{ easy_rsa_dir }}/pki/private/server.key"
        dest: "/etc/openvpn/server.key"
        remote_src: yes
      become: yes

    - name: Copy Diffie-Hellman parameters to OpenVPN directory
      copy:
        src: "{{ easy_rsa_dir }}/pki/dh.pem"
        dest: "/etc/openvpn/dh.pem"
        remote_src: yes
      become: yes

    - name: Copy HMAC key to OpenVPN directory
      copy:
        src: "{{ easy_rsa_dir }}/ta.key"
        dest: "/etc/openvpn/ta.key"
        remote_src: yes
      become: yes
      when: gen_hmac.changed

    - name: Create OpenVPN server configuration
      copy:
        dest: "/etc/openvpn/server.conf"
        content: |
          port {{ openvpn_port }}
          proto {{ openvpn_protocol }}
          dev tun
          ca ca.crt
          cert server.crt
          key server.key
          dh dh.pem
          server {{ openvpn_server_ip }} {{ openvpn_subnet_mask }}
          ifconfig-pool-persist ipp.txt
          push "redirect-gateway def1 bypass-dhcp"
          push "dhcp-option DNS {{ openvpn_dns1 }}"
          push "dhcp-option DNS {{ openvpn_dns2 }}"
          keepalive 10 120
          tls-auth ta.key 0
          cipher AES-256-CBC
          auth SHA256
          user nobody
          group nogroup
          persist-key
          persist-tun
          status openvpn-status.log
          verb 3
      become: yes
      notify:
        - Restart OpenVPN

    - name: Enable IP forwarding
      sysctl:
        name: net.ipv4.ip_forward
        value: "1"
        state: present
        reload: yes
      become: yes

    - name: Configure NAT using iptables
      command: iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -o eth0 -j MASQUERADE
      become: yes
      args:
        creates: "/etc/iptables/rules.v4"

    - name: Pre-seed debconf answers for iptables-persistent
      debconf:
        name: iptables-persistent
        question: "iptables-persistent/autosave_v4"
        value: "yes"
        vtype: "string"
      become: yes

    - name: Install iptables-persistent to save NAT settings
      apt:
        name: iptables-persistent
        state: present
        install_recommends: no
      become: yes

    - name: Save iptables rules
      command: netfilter-persistent save
      become: yes

    - name: Start and enable OpenVPN service
      systemd:
        name: openvpn@server
        enabled: yes
        state: started
      become: yes

  handlers:
    - name: Restart OpenVPN
      systemd:
        name: openvpn@server
        state: restarted
      become: yes

""", language='yaml')

st.write("For the full playbook, refer to the source code at the top of this blog.")

st.header("4. Create the Inventory File")
st.write("""
The **inventory file** tells Ansible which server to connect to and how. Here’s how to create it:
""")
st.code("""
# Create the hosts file
touch hosts

# Add the following content:
[vpn]
your-server-ip ansible_ssh_user=ubuntu ansible_ssh_private_key_file=/path/to/private/key.pem
""", language='ini')

st.header("5. Running the Playbook")
st.write("""
Now, run the playbook to set up OpenVPN on your server:
""")
st.code("""
ansible-playbook -i hosts openvpn_setup.yml
""", language='bash')

st.header("6. Troubleshooting")
st.write("""
- **Permission denied (publickey)**: Ensure the path to your private key is correct.
- **Ansible not found**: Ensure Ansible is installed by running `ansible --version`.
""")
