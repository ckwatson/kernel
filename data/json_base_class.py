#
# Json_base_class.py
#
"""module for the class Json_base"""

import os
import json


# base exception
class Data_exception(OSError):
    pass


# cache specific exception
class Cache_exception(Data_exception):
    pass


# file specific exception
class File_exception(Data_exception):
    pass


class virutal_JSON_Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)


class Json_base:
    """The parent virtual class for any data object using JSON serialization"""

    # default file properties
    # all files start with Data, this could be changed
    file_prefix = os.path.join(
        os.getcwd(),
        "Data",
    )
    # this suffix should be overwritten by children of this class
    file_suffix = ".virtual"

    @staticmethod
    def test():
        return True

    @classmethod
    def default_object(cls):
        return cls()

    @classmethod  # this function returns the number of instances of the reaction_mechanism, it is called by the reaction_mechanism template
    def get_number_of_instances(cls_obj):
        return cls_obj.__number_of_instances_of_self

    # requires that prepare_load returns a dictionary with attributes and their associated new values
    @classmethod
    def load_object(
        cls, data_identifier, json_dictionary=None, file_path=None, cache_path=None
    ):
        try:
            loaded_object = cls.default_object()
            if json_dictionary == None:
                print(data_identifier, file_path)
                loaded_object.load(data_identifier, file_path)
            else:
                for key, value in loaded_object.prepare_load(
                    loaded_dict=json_dictionary
                ).items():
                    setattr(loaded_object, key, value)
                loaded_object.update()
        except IOError as e:
            # handle any IO issues
            print("ERROR " + cls.__name__ + "CLASSLOAD FAILED")
            raise
        else:
            # return the default object
            return loaded_object

    def __init__(self, *args, **kw_args):
        if ("JSON_Encoder" in self.__dict__) and self.JSON_Encoder is not None:
            pass  # already has a JSON enocder implemented
        else:
            self.JSON_Encoder = virutal_JSON_Encoder

        # update is called when the object is instantiated
        self.update()

    def __eq__(self, other):
        return isinstance(other, Json_base)

    def __ne__(self, other):
        return not (self == other)

    # any code that happens just before the object is deleted belongs in this function
    def __del__(self):
        pass

    # this method  is invoked when str() is called on an Json_base object
    def __str__(self):
        if type(self).__name__ is not Json_base.__name__:
            return (
                "This is "
                + type(self).__name__
                + " which is a child of the Json_base class and has not implemented the __str__() function yet"
            )
        else:
            return "__str__() This is an instance of a Json_base object"

    # this method  is invoked when repr() is called on an Json_base object
    def __repr__(self):
        if type(self).__name__ is not Json_base.__name__:
            return (
                "This is "
                + type(self).__name__
                + " which is a child of the Json_base class and has not implemented the __repr__() function yet"
            )
        else:
            return "__repr__() This is an instance of a Json_base object"

    def __contains__(self, value):
        return NotImplemented

    # this function is overloaded by subclasses
    def prepare_store(self):
        if type(self).__name__ is not Json_base.__name__:
            return (
                "This is "
                + type(self).__name__
                + " which is a child of the Json_base class and has not implemented the prepare_store() function yet"
            )
        else:
            return "prepare_store() This is an instance of a Json_base object"

    # this function is overloaded by subclasses
    def prepare_load(self, loaded_dict={}):
        if type(self).__name__ is not Json_base.__name__:
            return (
                "This is "
                + type(self).__name__
                + " which is a child of the Json_base class and has not implemented the prepare_load() function yet"
            )
        else:
            return "prepare_load() This is an instance of a Json_base object"

    # requires that prepare_load returns a dictionary with attributes and their associated new values
    def store(self, data_identifier, file_path=None, cache_path=None):
        try:
            if file_path is None and cache_path is None:
                # if no storage path is provided then we have a default implementation,
                # this is an implementation detail that needs to be ironed out
                # this is used only for 'default' storage, currently for the test functions
                target_path = os.path.join(
                    type(self).file_prefix, data_identifier + type(self).file_suffix
                )
                target_file = open(target_path, mode="w", encoding="UTF8")
                target_file.write(
                    json.dumps(self.prepare_store(), cls=self.JSON_Encoder)
                )
                target_file.close()

            elif file_path is None:
                pass
                # we are storing data in the cache of the browser
                #
                # === IMPLEMENT CACHE STORING HERE ===
                #
                # target_object = create()
                # target_object.write(json.dumps(self.prepare_store(), cls = self.JSON_Encoder))
                # target_object.close()
                #
                # === IMPLEMENT CACHE STORING HERE ===

            elif cache_path is None:
                # we are storing data in a file
                target_path = os.path.abspath(file_path)
                target_file = open(target_path, mode="w", encoding="UTF8")
                target_file.write(
                    json.dumps(self.prepare_store(), cls=self.JSON_Encoder)
                )
                target_file.close()

            else:  # both a cache and file path are provided - this is not correct operating procedure
                raise Data_exception(0, "No path to store data was provided")

        except Cache_exception as err:
            # handle any IO issues
            print(
                "ERROR "
                + type(self).__name__
                + " STORE FAILED - Implement error handling"
            )
            #
            # === IMPLEMENT CACHE ERROR HANDLING HERE ===
            #
            # === IMPLEMENT CACHE ERROR HANDLING HERE ===
            #
            print(err, err.args, sep="\n")  # default
            return False

        except File_exception as err:
            # handle any IO issues
            print("ERROR " + type(self).__name__ + " STORE FAILED")
            target_file.close()
            print(err, err.args, sep="\n")
            return False

        except Data_exception as err:
            # handle any IO issues
            print("ERROR " + type(self).__name__ + " STORE FAILED")
            print(err, err.args, sep="\n")
            return False

        # generic catch
        except OSError as oserr:
            print(oserr, oserr.args, sep="\n")
            quit(0)
        else:
            return True

    # requires that prepare_load returns a dictionary with attributes and their associated new values
    def load(self, data_identifier, file_path=None, cache_path=None):
        try:
            if file_path is None and cache_path is None:
                # if no storage path is provided then we have a default implementation,
                # this is an implementation detail that needs to be ironed out
                # TEMPORARY SOLUTION
                target_path = os.path.join(
                    type(self).file_prefix, data_identifier + type(self).file_suffix
                )
                target_file = open(target_path, mode="r", encoding="UTF8")
                for key, value in self.prepare_load(
                    loaded_dict=json.loads(target_file.readline())
                ).items():
                    setattr(self, key, value)
                self.update()
                target_file.close()

            elif file_path is None:
                pass
                # we are storing data in the cache of the browser
                #
                # === IMPLEMENT CACHE STORING HERE ===
                #
                # target_path = modify(cache_path)
                # target_cache_object = open()
                # for key, value in self.prepare_load(loaded_dict = json.loads(target_cache_object.readline())).items():
                #     setattr(self, key, value)
                # self.update()
                #
                # === IMPLEMENT CACHE STORING HERE ===

            elif cache_path is None:
                # we are storing data in a file
                target_path = os.path.abspath(file_path)
                target_file = open(target_path, mode="r", encoding="UTF8")
                for key, value in self.prepare_load(
                    loaded_dict=json.loads(target_file.readline())
                ).items():
                    setattr(self, key, value)
                self.update()
                target_file.close()

            else:  # both a cache and file path are provided - this is not correct operating procedure
                raise data_exception(0, "No path to load data was provided")

        except Cache_exception as err:
            # handle any IO issues
            print(
                "ERROR "
                + type(self).__name__
                + " LOAD FAILED - Implement error handling"
            )
            #
            # === IMPLEMENT CACHE ERROR HANDLING HERE ===
            #
            # === IMPLEMENT CACHE ERROR HANDLING HERE ===
            #
            print(err, err.args, sep="\n")  # default
            raise

        except File_exception as err:
            # handle any IO issues
            print("ERROR " + type(self).__name__ + " LOAD FAILED")
            print(err, err.args, sep="\n")
            raise

        except Data_exception as err:
            # handle any IO issues
            print("ERROR " + type(self).__name__ + " LOAD FAILED")
            print(err, err.args, sep="\n")
            raise

        # generic catch
        except OSError as oserr:
            print(oserr, oserr.args, sep="\n")
            quit(0)
        else:
            return True

    def update(self):
        if type(self).__name__ is not Json_base.__name__:
            print(
                "This is "
                + type(self).__name__
                + " which is a child of the Json_base class and has not implemented the update() function yet"
            )
        else:
            print("update) This is an instance of a Json_base object")


if __name__ == "__main__":
    print("This is the Json_base_class module, it is a virtual class")
    Json_base.test()
