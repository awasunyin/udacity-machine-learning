import pickle
from get_data import getData


def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if all_messages != 'NaN':
        fraction = float(poi_messages) / float(all_messages)
    return fraction


data_dict = getData()

submit_dict = {}
for name in data_dict:
    data_point = data_dict[name]

    print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    print fraction_to_poi
    submit_dict[name] = {"from_poi_to_this_person": fraction_from_poi,
                         "from_this_person_to_poi": fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi


#####################

def submitDict():
    return submit_dict

"""We take a couple of lessons from this:

Anyone can make mistakes--be skeptical of 
your results!
100% accuracy should generally make you suspicious. 
Extraordinary claims require extraordinary proof.
If there's a feature that tracks your labels a little 
too closely, it's very likely a bug!
If you're sure it's not a bug, you probably don't need 
machine learning--you can just use that feature alone to 
assign labels."""

"""There are two big univariate feature selection tools in 
sklearn: SelectPercentile and SelectKBest. The difference is 
pretty apparent by the names: SelectPercentile selects the X% 
of features that are most powerful (where X is a parameter) and 
SelectKBest selects the K features that are most powerful (where 
K is a parameter).

"""