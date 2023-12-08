# https://www.kaggle.com/code/awater1223/op2-00-basic-metadata-eda#Data-split


control_ids = [
    #     'LSM-36361'  # DMSO
    "LSM-43181",  # Belinostat
    "LSM-6303",  # Dabrafenib
]

privte_ids = [
    "LSM-45710",
    "LSM-4062",
    "LSM-2193",  #  'forskolin' -> 'Colforsin'
    "LSM-4105",
    "LSM-4031",
    "LSM-1099",
    "LSM-45153",
    "LSM-3822",
    "LSM-4933",
    "LSM-45630",  # 'KD-025' -> 'SLx-2119'
    "LSM-6258",
    "LSM-1023",
    "LSM-2655",
    "LSM-47602",
    "LSM-3349",
    "LSM-1020",
    "LSM-1143",
    "LSM-3828",
    "LSM-1051",
    "LSM-1120",
    "LSM-5467",
    "LSM-2292",
    "LSM-43293",
    "LSM-45437",
    "LSM-2703",
    "LSM-45831",
    "LSM-1179",
    "LSM-1199",
    "LSM-1190",
    "LSM-36374",
    "LSM-5215",
    "LSM-1195",
    "LSM-45468",
    "LSM-45410",
    "LSM-47459",
    "LSM-45663",
    "LSM-45518",
    "LSM-1062",
    "LSM-3667",  # 'BRD-K74305673' -> 'IMD-0354',
    "LSM-1032",
    "LSM-5855",
    "LSM-45988",
    "LSM-24954",  # 'BRD-K98039984' -> 'Prednisolone'
    "LSM-6286",
    "LSM-45984",
    "LSM-1124",
    "LSM-1165",
    "LSM-42802",
    "LSM-1121",
    "LSM-6308",
    "LSM-1136",
    "LSM-1186",
    "LSM-45915",
    "LSM-2621",
    "LSM-5341",
    "LSM-45724",
    "LSM-2219",
    "LSM-2936",
    "LSM-3171",
    "LSM-46889",
    "LSM-2379",
    "LSM-47132",
    "LSM-47120",
    "LSM-47437",
    "LSM-1139",
    "LSM-1144",
    "LSM-4353",
    "LSM-1210",
    "LSM-5887",
    "LSM-1025",
    "LSM-5771",
    "LSM-1132",
    "LSM-1263",  # 'BRD-A04553218' -> 'Chlorpheniramine'
    "LSM-1167",
    "LSM-1194",  # 'BRD-A92800748' -> 'TIE2 Kinase Inhibitor'
    "LSM-45948",
    "LSM-45514",
    "LSM-5430",
    "LSM-2309",
]

public_ids = [
    "LSM-43216",
    "LSM-1050",
    "LSM-45849",
    "LSM-42800",
    "LSM-1131",
    "LSM-6335",
    "LSM-1211",
    "LSM-45239",
    "LSM-1130",
    "LSM-45786",
    "LSM-5199",
    "LSM-45281",
    "LSM-6324",  # 'ACY-1215' -> 'Ricolinostat'
    "LSM-3309",
    "LSM-1056",
    "LSM-45591",
    "LSM-46203",
    "LSM-5662",
    "LSM-47134",  # 'SB-2342' -> '5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine	'
    "LSM-45637",
    "LSM-1127",
    "LSM-46971",
    "LSM-1172",
    "LSM-46042",
    "LSM-1101",
    "LSM-45758",
    "LSM-5218",
    "LSM-2287",
    "LSM-1014",
    "LSM-1040",  #  'fostamatinib' -> 'Tamatinib'
    "LSM-1476;LSM-5290",
    "LSM-45680",  # 'basimglurant' -> 'RG7090'
    "LSM-4349",  # '5-iodotubercidin' -> 'IN1451'
    "LSM-3425",
    "LSM-45806",
    "LSM-45616",  # 'SB-683698' -> 'TR-14035'
    "LSM-1055",
    "LSM-43281",  # 'C-646' -> 'STK219801'
    "LSM-5690",
    "LSM-1155",
    "LSM-2499",
    "LSM-2382",  # 'JTC-801' -> 'UNII-BXU45ZH6LI'
    "LSM-45220",
    "LSM-1037",
    "LSM-1005",
    "LSM-1180",
    "LSM-36812",
    "LSM-45924",  # 'filgotinib' -> 'GLPG0634'
    "LSM-2013",  # 'TL-HRAS-61' -> TL_HRAS26'
    "LSM-4738",
]
