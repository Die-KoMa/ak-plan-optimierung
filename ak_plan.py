import sys
import json
import csv
import gurobipy as gp
from gurobipy import GRB

def read(json_input_file_name, csv_input_file_name):
    with open(json_input_file_name, 'r') as json_file:
        data = json.load(json_file)
        room_dict = { room["name"] : room for room in  data["rooms"]}
        slot_dict = { slot["name"] : slot for slot in  data["slots"]}
        ak_dict = { ak["name"] : ak for ak in  data["aks"]}

        #add blacklists:
        for ak in ak_dict.values():
            if "blacklist" not in ak:
                ak["blacklist"]=list()

    with open(csv_input_file_name, newline='') as csv_file:
        reader = csv.reader(csv_file)
        person_dict = {}
        for row in reader:
            if reader.line_num==1:
                ak_list=row #al_list[0]=="name" or empty
            else:
                name="person"+str(reader.line_num-2)+"_"+row[0]
                attend=set()
                lead=set() #
                for i,entry in enumerate(row[1:], 1):
                    if entry=="1":
                        attend.add(ak_list[i])
                    elif entry=="2":
                        attend.add(ak_list[i])
                        lead.add(ak_list[i])

                person_dict[name]={"name":name,
                                    "attend":attend,
                                    "lead": lead }

    return room_dict, slot_dict, ak_dict, person_dict

def solve(room_dict, slot_dict, ak_dict, person_dict):
    model = gp.Model("akplan")

    #S*R - rooms available in slot
    available=[(slot_name,room_name) for slot_name, slot in slot_dict.items()
                for room_name in room_dict if room_name in slot["rooms"]]

    #Create varibles: S*R*A
    x=model.addVars(available,ak_dict.keys(),vtype=GRB.BINARY,name="x")

    #Constraints:

    #nur ein ein AK pro Raum und Slot
    model.addConstrs(
        (x.sum(s,r,'*') <=1 for s,r in available ) , "raum_ak_ueberbuchung")

    #jeder AK genau einmal:
    model.addConstrs(
        (x.sum('*','*',a) ==1 for a in ak_dict ) , "ak_stattfinden")

    #AK-Leitung darf sich nicht ueberschneiden:
    for person in person_dict.values():
        for s,slot in slot_dict.items():
            model.addConstr( sum(x[s,r,a] for r in slot["rooms"]
                                        for a in person["lead"] ) <=1,
                            f"ak_leitung_{person['name']}_{s}")
    #todo: kann man das Looping hier irgendwie vermeiden?

    #Raumkapazitaet:
    for s,r in available:
            model.addConstr( sum(x[s,r,a] for person in person_dict.values()
                                        for a in person["attend"] ) <=room_dict[r]["capacity"],
                            f"raumkapazitaet_{s}_{r}")


    #reset upper bounds
    #Reso-AK nur in resoable-Slots
    for s,slot in slot_dict.items():
        for a,ak in ak_dict.items():
            if ak["reso"] and not slot["resoable"]:
                for r in slot["rooms"]:
                    x[s,r,a].ub=0

    #AK Blacklist
    for a,ak in ak_dict.items():
        for s in ak["blacklist"]:
            for r in slot_dict[s]["rooms"]:
                x[s,r,a].ub=0

    #Ueberschneidungs-Variablen:
    rel_coeffs={(s,p): 1/len(person["attend"]) for s in slot_dict for p,person in person_dict.items()}

    overlap=model.addVars(slot_dict,person_dict,vtype=GRB.INTEGER,name="overlap")

    #wollen den Overlap gering halten
    # d.h es soll maximal 1 gleichzeitg belgegt werden
    # d.h. overlap_(s,p)=max{#AKs die p in s besucht - 1 , 0 } >= #AKs die p in s besucht - 1
    for s,slot in slot_dict.items():
        for p,person in person_dict.items():
            model.addConstr(
                sum(x[s,r,a] for r in slot["rooms"] for a in person["attend"] ) - overlap[s,p]<=1 ,
                f"ueberscheidung_{s}_{p}" )


    #Maximale relative Ueberschneidungs Variable:
    max_rel_overlap=model.addVar(name="max_rel_overlap")

    #Contraint: >= realtive Ueberschneidungen
    #=> bei Minimierung wird es das Maximum der rel. Uberschneidungen
    model.addConstrs(
        (overlap.prod(rel_coeffs,'*',p)- max_rel_overlap <=0  for p in person_dict ),
        ">=rel_uberschneidung")

    #Summe relative Ueberschneidungs Variable:
    sum_rel_overlap=model.addVar(name="sum_rel_overlap")

    #Contraint: == Summe aller realtive Ueberschneidungen
    model.addConstr(
        overlap.prod(rel_coeffs)- sum_rel_overlap ==0 ,
        "summe rel_uberschneidung")


    # Set objective
#    model.setObjective( sum_rel_overlap + 1*len(person_dict)*max_rel_overlap , GRB.MINIMIZE )
    model.setObjective( sum_rel_overlap + 1/2*len(person_dict)*max_rel_overlap , GRB.MINIMIZE )

    # Compute optimal solution
    model.optimize()

    #print solution
    if model.status == GRB.OPTIMAL:
        solution=model.getAttr('x',x)
        for s,r in available:
            for a in ak_dict:
                if solution[s,r,a]> 0.5:
                    print(f"{s}, {r}, {a} : {solution[s,r,a]}")

    print("max_rel_overlap<= ", max_rel_overlap.x)
    print("sum_rel_overlap: ", sum_rel_overlap.x)

    for p in person_dict:
        for s in slot_dict:
            if overlap[s,p].x >0.5:
                print(f"Uberschneidungen von {p} in {s}: ",overlap[s,p].x)






def _main():
    if len(sys.argv)>=3:
        json_input_file_name=sys.argv[1]
        csv_input_file_name=sys.argv[2]
    else:
        json_input_file_name=input("JSON Datei:")
        csv_input_file_name=input("CSV Datei:")

    room_dict, slot_dict, ak_dict, person_dict=read(json_input_file_name, csv_input_file_name)

    solve(room_dict, slot_dict, ak_dict, person_dict)








if __name__ == '__main__':
    _main()