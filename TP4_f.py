'''
Created on 28  Fev 2022
@author: menez

Syntax analysis of the file of frames 
'''

import binascii
import struct # pour unpack
#====================================================
dico_ethertypes = {"0x800" : "Internet Protocol version 4 (IPv4)", "0x0806" : "Address Resolution Protocol (ARP)", "0x0842" : "Wake-on-LAN1", "0x22F3" : "IETF TRILL Protocol", "0x6003" : "DECnet Phase IV", "0x8035" : "Reverse Address Resolution Protocol (RARP)", "0x809b" : "AppleTalk (Ethertalk)", "0x80F3" : "AppleTalk Address Resolution Protocol (AARP)", "0x8100" : "VLAN-tagged frame (IEEE 802.1Q) & Shortest Path Bridging IEEE 802.1aq2", "0x8137" : "Novell IPX (alternatif)", "0x8138" : "Novell", "0x8204" : "QNX Qnet", "0x86dd" : "Internet Protocol, Version 6 (IPv6)", "0x8808" : "Ethernet flow control", "0x8809" : "Slow Protocols (IEEE 802.3)", "0x8819" : "CobraNet", "0x8847" : "MPLS unicast", "0x8848" : "MPLS multicast", "0x8863" : "PPPoE Discovery Stage", "0x8864" : "PPPoE Session Stage", "0x8870" : "Jumbo Frames", "0x887B" : "HomePlug 1.0 MME", "0x888E" : "EAP over LAN (IEEE 802.1X)", "0x8892" : "Profinet RT", "0x8896" : "Ethersound", "0x889A" : "HyperSCSI (SCSI over Ethernet)", "0x88A2" : "ATA over Ethernet", "0x88A4" : "EtherCAT Protocol", "0x88A8" : "Provider Bridging (IEEE 802.1ad) & Shortest Path Bridging IEEE 802.1aq3", "0x88AB" : "Powerlink", "0x88CC" : "Link Layer Discovery Protocol (LLDP)", "0x88CD" : "Sercos", "0x88E1" : "HomePlug AV MME[citation nécessaire]", "0x88E3" : "Media Redundancy Protocol (IEC62439-2)", "0x88E5" : "MAC security (IEEE 802.1ae)", "0x88F7" : "Precision Time Protocol (IEEE 1588)", "0x8902" : "IEEE 802.1ag Connectivity Fault Management (CFM) Protocol / ITU-T Recommendation Y.1731 (OAM)", "0x8906" : "Fibre Channel over Ethernet (FCoE)", "0x8914" : "FCoE Initialization Protocol", "0x8915" : "RDMA over Converged Ethernet (RoCE)", "0x9000" : "Configuration Testing Protocol (Loop)4, utilisé notamment pour les keepalives Ethernet chez Cisco5", "0x9100" : "Q-in-Q", "0xCAFE" : "Veritas Low Latency Transport (LLT)6 for Veritas Cluster Server"}

protocoles_ip = {'HOPOPT': 0, 'ICMP': 1, 'IGMP': 2, 'GGP': 3, 'IPv4': 4, 'ST': 5, 'TCP': 6, 'CBT': 7, 'EGP': 8, 'IGP': 9, 'BBN-RCC-MON': 10, 'NVP-II': 11, 'PUP': 12, 'ARGUS (deprecated)': 13, 'EMCON': 14, 'XNET': 15, 'CHAOS': 16, 'UDP': 17, 'MUX': 18, 'DCN-MEAS': 19, 'HMP': 20, 'PRM': 21, 'XNS-IDP': 22, 'TRUNK-1': 23, 'TRUNK-2': 24, 'LEAF-1': 25, 'LEAF-2': 26, 'RDP': 27, 'IRTP': 28, 'ISO-TP4': 29, 'NETBLT': 30, 'MFE-NSP': 31, 'MERIT-INP': 32, 'DCCP': 33, '3PC': 34, 'IDPR': 35, 'XTP': 36, 'DDP': 37, 'IDPR-CMTP': 38, 'TP++': 39, 'IL': 40, 'IPv6': 41, 'SDRP': 42, 'IPv6-Route': 43, 'IPv6-Frag': 44, 'IDRP': 45, 'RSVP': 46, 'GRE': 47, 'DSR': 48, 'BNA': 49, 'ESP': 50, 'AH': 51, 'I-NLSP': 52, 'SWIPE (deprecated)': 53, 'NARP': 54, 'MOBILE': 55, 'TLSP': 56, 'SKIP': 57, 'IPv6-ICMP': 58, 'IPv6-NoNxt': 59, 'IPv6-Opts': 60, '': 254, 'CFTP': 62, 'SAT-EXPAK': 64, 'KRYPTOLAN': 65, 'RVD': 66, 'IPPC': 67, 'SAT-MON': 69, 'VISA': 70, 'IPCV': 71, 'CPNX': 72, 'CPHB': 73, 'WSN': 74, 'PVP': 75, 'BR-SAT-MON': 76, 'SUN-ND': 77, 'WB-MON': 78, 'WB-EXPAK': 79, 'ISO-IP': 80, 'VMTP': 81, 'SECURE-VMTP': 82, 'VINES': 83, 'IPTM': 84, 'NSFNET-IGP': 85, 'DGP': 86, 'TCF': 87, 'EIGRP': 88, 'OSPFIGP': 89, 'Sprite-RPC': 90, 'LARP': 91, 'MTP': 92, 'AX.25': 93, 'IPIP': 94, 'MICP (deprecated)': 95, 'SCC-SP': 96, 'ETHERIP': 97, 'ENCAP': 98, 'GMTP': 100, 'IFMP': 101, 'PNNI': 102, 'PIM': 103, 'ARIS': 104, 'SCPS': 105, 'QNX': 106, 'A/N': 107, 'IPComp': 108, 'SNP': 109, 'Compaq-Peer': 110, 'IPX-in-IP': 111, 'VRRP': 112, 'PGM': 113, 'L2TP': 115, 'DDX': 116, 'IATP': 117, 'STP': 118, 'SRP': 119, 'UTI': 120, 'SMP': 121, 'SM (deprecated)': 122, 'PTP': 123, 'ISIS over IPv4': 124, 'FIRE': 125, 'CRTP': 126, 'CRUDP': 127, 'SSCOPMCE': 128, 'IPLT': 129, 'SPS': 130, 'PIPE': 131, 'SCTP': 132, 'FC': 133, 'RSVP-E2E-IGNORE': 134, 'Mobility Header': 135, 'UDPLite': 136, 'MPLS-in-IP': 137, 'manet': 138, 'HIP': 139, 'Shim6': 140, 'WESP': 141, 'ROHC': 142, 'Ethernet': 143, 'AGGFRAG': 144, 'Reservé': 255}

types_icmp = {0: 'Reponse echo', 3: 'Destinataire inaccessible', 4: 'Extinction source', 5: 'Redirection', 8: "Demande d'echo", 11: 'Temps depasse', 12: 'En tete errone', 13: 'Demande heure', 14: 'Reponse heure', 15: 'Demande adresse IP', 16: 'Reponse adresse IP', 17: 'Demande masque sous-reseau', 18: 'Reponse masque sous-reseau'}

def readtrames(filename):
	"""
	Cette fonction fabrique une liste de chaines de caracteres a partir du 
	fichier contenant les trames.
	
	Chaque chaine de la liste rendue est une trame du fichier.
	
	return : liste des trames contenues dans le fichier
	"""
	file = open(filename)
	lestrames = [] # List of frames (= lestrames)
	trame = ""  # Current frame .. string vide
	i = 1
	for line in file : # acces au fichier ligne par ligne
		line = line.rstrip('\n') # on enleve le retour chariot de la ligne
		line = line[5:53]        # on ne garde que les colonnes interessantes  
		print ("ligne {} : {}".format(i,line))
		trame = trame + line

		if (len(line) == 0): # Trame separator
			
			# On enregistre la trame dans lestrames
			trame = trame.replace(' ','') # on enleve les blancs
			lestrames.append(trame) # on ajoute la trame a la liste 
			trame = ""       # reset trame
		i = i+1
		
	# Si a la fin du fichier, il reste une trame à enregister 
	if len(trame) != 0 : # Last frame
		trame = trame.replace(' ','') # on enleve les blancs
		lestrames.append(trame) # on ajoute la trame a la liste 

	return lestrames

def unhexlify_lestrames(l):
	''' l : liste de trames '''
	for i,trame in enumerate(l):
		print("{}({})".format(type(trame),len(trame)),end='')
		l[i] = binascii.unhexlify(trame) 
		print("   --> {}({}) : {} ...".format(type(trame), len(trame), trame[0:10]))

def analyse_syntaxique(filename) :
	''' Analyse syntaxique du fichier filename'''

	# Transformation des echanges contenus dans le fichier
	# vers une liste de strings :  une string = une trame
	print("="*60)
	trames = readtrames(filename)

	print("="*60)
	print("\nTrames lues depuis le fichier :\n")
	for i,t in enumerate(trames):
		print("trame #{} : {}".format(i,t))
	# Unhexlify la liste de trames
	unhexlify_lestrames(trames)
	return trames

def analyse_semantique(trame):
	"""
	Analyse une trame Ethernet :  cf https://fr.wikipedia.org/wiki/Ethernet    
	Input : trame est un tableau d'octets
	"""
	print("-"*60)
	print ("\nTrame Ethernet en cours d'analyse ({}): \n{}".format( type(trame), trame))
  
	# Analyse du header ETHERNET
	eth_header = trame[0:14]
	eth_mac_dest, eth_mac_src, eth_type = struct.unpack('!6s6sH' , eth_header)
	
	print ('Destination MAC : {}'.format(eth_mac_dest.hex(":")))
	print ('Source MAC \t: {}'.format(eth_mac_src.hex(":")))
	print ('Type:\t\t: {} {}'.format(hex(eth_type), (dico_ethertypes[hex(eth_type)])))

	if(hex(eth_type) == "0x800"): # type IP
		ip_header = struct.unpack('!BBHHHBBH4s4s', trame[14:34])
		# Version + IHL	|  TOS	| Total Length | Identification | Flags + Fragment offset | TTL | Protocole | Header Checksum | Adresse source  | Adresse destination|
		#	     B 		| 	B 	| 		H 	   | 		H 		| 			H 			  |  B  |	  B 	| 	    	H 	  | 	4s 			| 		  4s		 | B -> 1 octet en int, H-> 2 octets en int, s -> sequence d'octets
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		# |Version|  IHL  |Type of Service|          Total Length         |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+  
		# |         Identification        |Flags|      Fragment Offset    |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		# |  Time to Live |    Protocol   |         Header Checksum       |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		# |                       Source Address                          |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		# |                    Destination Address                        |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		# |                    Options                    |    Padding    |
		# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		version_ihl = ip_header[0]
		version = version_ihl >> 4 #la version correspond au 4 premiers bits du premier octet
		ihl = version_ihl & 0xF #l'ihl correspond au 4 derniers bits du premier octet

		iph_length = ihl * 4 #Longueur de l'en tete donnée en mots de 32 bits soit 4 octets => donne la taille en octets de l'en tete IP

		ttl = ip_header[5]
		protocol = ip_header[6]
		adresse_source = ('.'.join(f'{c}' for c in ip_header[8]))
		adresse_dest = ('.'.join(f'{c}' for c in ip_header[9]))
		print(f"Version IP: {str(version)}, Taille de l'en-tete IP: {str(iph_length)}, TTL : {str(ttl)}, Protocole {str(protocol)}, Adresse source: {str(adresse_source)}, Adresse destination: {str(adresse_dest)}")
		t = 14 + iph_length
		if(protocol == protocoles_ip["UDP"]): #UDP
			udp_header = struct.unpack('!HHHH', trame[t:t+8]) #header UDP taille 8 octets
			port_entree = udp_header[0]
			port_sortie = udp_header[1]
			udp_len =  udp_header[2]
			udp_checksum =  hex(udp_header[3])
			print(f"Port d'entrée: {str(port_entree)}, Port de sortie: {str(port_sortie)}, Longueur en-tête : {str(udp_len)}, Checksum {str(udp_checksum)}")
			data = trame[t+8:]
			print(f"Données UDP: {data}")
		elif(protocol == protocoles_ip["TCP"]): # TCP
			print("TCP")
			tcp_header = struct.unpack('!HH4s4sBBHHH', trame[t:t+20]) #header TCP taille 20 octets (32bits * 5)
			port_entree = tcp_header[0]
			port_sortie = tcp_header[1]
			no_seq = tcp_header[2].hex()
			no_ack = tcp_header[3].hex()
			tcp_len = tcp_header[4] >> 4
			tcp_len = tcp_len*4 #la taille de l'en tete TCP est donnée en mots de 32 bits = 4 octets
			tcp_checksum = hex(tcp_header[7])
			print(f"Port d'entrée: {str(port_entree)}, Port de sortie: {str(port_sortie)}, Longueur en-tête : {str(tcp_len)}, Checksum {str(tcp_checksum)}, Numéro de séquence {no_seq}, Numéro ACK {no_ack}")
			data = trame[t+20:]
			print(f"Données TCP: {data}")
		elif(protocol == protocoles_ip["ICMP"]): # ICMP
			print("ICMP")
			icmp_header = struct.unpack('!BBH', trame[t:t+4]) #Header ICMP taille 8 octets dont 4 pour le type + code + checksum
			icmp_data = trame[t+4:]
			icmp_type = icmp_header[0]
			icmp_code = icmp_header[1]
			icmp_checksum = icmp_header[2]
			print(f"Type: {icmp_type} => {types_icmp[icmp_type]}\nCode: {icmp_code}\nChecksum: {hex(icmp_checksum)}")
			print(f"Donnees ICMP: {icmp_data}")
#=================================================================
if __name__ == '__main__':

	filename = "trames.txt"
	lestrames = analyse_syntaxique(filename)
	# Analyse sémantique 
	for trame in lestrames:
		analyse_semantique(trame)