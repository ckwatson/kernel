#
# molecular_species.py
#
"""module for the class molecular_species"""

import os
import sys
import json
from . import json_base_class


class mol_JSON_Encoder(json.JSONEncoder):
    def default(self, obj):
        # print(obj, file=sys.stderr)
        if isinstance(obj, dict) and isinstance(obj.values()[0], molecular_species):
            return [molecule.name for molecule in iter(obj)]
        if isinstance(obj, list) and isinstance(obj[0], molecular_species):
            return [molecule.name for molecule in obj]
        if isinstance(obj, molecular_species):
            return obj.name
        # Let the base class default method raise the TypeError

        return json.JSONEncoder.default(self, obj)


class molecular_species(json_base_class.Json_base):
    """The molecular species class"""

    # default file properties
    file_prefix = os.path.join(
        os.getcwd(),
        "Data",
        "Molecules",
    )
    file_suffix = ".mol"

    @staticmethod
    def test(quick_hack=False):
        if not quick_hack:
            number_of_tests = 4
            test = [False] * number_of_tests
            # run this test
            test[0] = True

            if test[0]:
                name = "H"
                energy = 0.0
                atom_list = {"H": 1}
            elif test[1]:
                name = "H2"
                energy = 0.0
                atom_list = {"H": 2}
            elif test[2]:
                name = "Br"
                energy = 0.0
                atom_list = {"Br": 1}
            elif test[3]:
                name = "Br2"
                energy = 0.0
                atom_list = {"Br": 2}
            else:
                pass

            test_molecule = molecular_species(name, energy, atom_list)
            test_molecule.store(name)
            new_molecule = molecular_species.load_object(name)
            if test_molecule == new_molecule:
                print(
                    "The reagent was successfully stored and loaded, no data was lost"
                )
                return True
            else:
                print(
                    "There was an error storing and loading the reagent, the objects are not identical"
                )
                print(new_molecule.__dict__, test_molecule.__dict__, sep="\n")
                return False
        elif quick_hack:
            species = [
                "O",
                "O2",
                "O3",
                "H",
                "H2",
                "Br",
                "Br2",
                "HBr",
                "OH",
                "OOH",
                "HOOH",
                "H2O",
            ]
            atom_list = [
                {"O": 1},
                {"O": 2},
                {"O": 3},
                {"H": 1},
                {"H": 2},
                {"Br": 1},
                {"Br": 2},
                {"Br": 1, "H": 1},
                {"O": 1, "H": 1},
                {"O": 2, "H": 1},
                {"O": 2, "H": 2},
                {"O": 1, "H": 2},
            ]
            val_list = [
                231.7,
                0.0,
                163.2,
                203.3,
                0.0,
                82.4,
                3.1,
                -53.4,
                34.2,
                22.6,
                -105.6,
                -228.6,
            ]
            Es = [float(val) * 1000 for val in val_list]  # use kJ

            for name, energy, atom_list in zip(species, Es, atom_list):
                test_mol = molecular_species(name, energy, atom_list)
                test_mol.store(name)
                new_mol = molecular_species.load_object(name)
                if test_mol == new_mol:
                    print(
                        "The reagent was successfully stored and loaded, no data was lost"
                    )
                    return True
                else:
                    print(
                        "There was an error storing and loading the reagent, the objects are not identical"
                    )
                    return False
                print(test_mol.__dict__, new_mol.__dict__, sep="\n")

    @classmethod
    def default_object(cls):
        return cls("H", 0.0, {"H": 1})

    name = None
    energy = None
    atom_list = None

    def __init__(
        self, molecule_name, gibbs_free_energy, atom_list=None, *args, **kw_args
    ):
        # handle molecule JSON serialization properly
        self.JSON_Encoder = mol_JSON_Encoder

        # properties of the molecule
        self.name = str(molecule_name)
        self.energy = float(gibbs_free_energy)
        self.atom_list = atom_list
        # now initialize the parent class "Json_base", which actually does not need to be initialzied:
        super().__init__(*args, **kw_args)

    def __eq__(self, other):
        return (
            isinstance(other, molecular_species)
            and self.name == other.name
            and self.energy == other.energy
            and self.atom_list == other.atom_list
            and super().__eq__(other)
        )

    # this method  is invoked when str() is called on an molecular_species class object
    def __str__(self):
        return str(self.prepare_store())

    # this method  is invoked when repr() is called on an molecular_species class object
    def __repr__(self):
        return str(self.prepare_store())

    # this function is overloaded by subclasses
    def prepare_store(self):
        prepared_dict = {
            "name": self.name,
            "energy": self.energy,
            "atom_list": self.atom_list,
        }
        return prepared_dict

    # this function is overloaded by subclasses
    def prepare_load(self, loaded_dict={}):
        loaded_dict["energy"] = float(loaded_dict["energy"])
        return loaded_dict

    def update(self):
        pass


if __name__ == "__main__":
    molecular_species.test()

    # end
