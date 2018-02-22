import re

import numpy as np

r = re.compile("([a-zA-Z]+)([0-9]+)?", re.I)


def engineer_deck(data_frame):
    """
    Some of the Cabin numbers we have contain multiple cabins and these
    could be on multiple decks. The lower a passenger had their cabin the
    less chance of survival, since the lowest cabins flooded first.
    Therefore we place these couple data points that span multiple decks
    in the lowest deck.

    :param data_frame: A dataframe with a Cabin attribute
    :return:
    """

    def get_deck(cabins):
        decks = []

        for cabin in cabins.split():  # cabins can be multiple, e.g. 'C27 C29'
            deck, room_number = r.match(cabin).groups()
            if deck not in decks:
                decks.append(deck)
        # sort the decks and return the lowest one.

        # NOTE; this creates a problem when since the Top deck (T) is
        # alphabetically 'below' the lowest deck (G). However, the odds of a
        # passenger booking both the top and bottom level decks are slim...
        return sorted(decks, reverse=True)[0]

    data_frame['Deck'] = data_frame.Cabin.apply(get_deck)

    return data_frame


def engineer_port(data_frame):
    """
    Depending on if a cabin was on the starboard or the port side of the
    ship could have had an impact on survivability, since the Titanic hit the
    iceberg more on one side than the other. Passengers that were assigned
    cabins on both sides of the ship we'll give an Unknown data point.

    :param data_frame: A dataframe with a Cabin attribute
    :return: a dataframe
    """

    def get_port(cabins):
        cabin_sides = []

        for cabin in cabins.split():  # cabins can be multiple, e.g. 'C27 C29'
            deck, room_number = r.match(cabin).groups()

            cabin_side = ''

            if room_number:
                cabin_side = 'P'

                if int(room_number) % 2 == 0:
                    # room number is even, cabin on starboard side.
                    cabin_side = 'S'

            if cabin_side and cabin_side not in cabin_sides:
                cabin_sides.append(cabin_side)

        if len(cabin_sides) != 1:
            # If there's no room number or if we have more than 1 side
            # mark cabin side unknown.
            cabin_sides = ['U']

        return cabin_sides[0]

    data_frame['Port'] = data_frame.Cabin.apply(get_port)

    return data_frame


def engineer_family_size(data_frame):
    # Amount of Siblings/Spouses + Parents/Children + yourself
    data_frame['FamilySize'] = data_frame.SibSp + data_frame.Parch + 1
    return data_frame


def engineer_title(data_frame):
    data_frame['Title'] = data_frame['Name'].str.split(
        ", ", expand=True)[1].str.split(".", expand=True)[0]
    return data_frame


def clean_uncommon_titles(data_frame):
    titles_to_clean = data_frame['Title'].value_counts() < 10
    data_frame.Title = data_frame.Title.apply(
        lambda title: 'Rare' if titles_to_clean.loc[title] else title)
    return data_frame


def clean_master_title(data_frame):
    """
    * Master's who are under 18 are only called such because they are young.
      I'll remove their title.
    * Masters above 18 hold some kind of rank. I'll categorise them as 'rank'

    :param data_frame: a pd.DataFrame
    :return: a pd.DataFrame with master titles cleaned
    """
    data_frame.loc[
        (data_frame.Title == 'Master') & (data_frame.Age < 18),
        'Title'
    ] = np.nan
    data_frame.loc[
        (data_frame.Title == 'Master') & (data_frame.Age > 18),
        'Title'
    ] = 'Rank'
    return data_frame


def clean_miss_title(data_frame):
    """
    * Miss's who are under 18 are called miss because they are women and
      nothing else. I'll remove their title.
    * Miss's who are above 18 have not been married, this makes them 'single'.

    :param data_frame: a pd.DataFrame
    :return: a pd.DataFrame with miss titles cleaned
    """
    data_frame.loc[
        (data_frame.Title == 'Miss') & (data_frame.Age < 18),
        'Title'
    ] = np.nan
    data_frame.loc[
        (data_frame.Title == 'Miss') & (data_frame.Age >= 18),
        'Title'
    ] = 'Single'
    return data_frame


def clean_mrs_title(data_frame):
    data_frame.loc[(data_frame.Title == 'Mrs'), 'Title'] = 'Married'
    return data_frame


def clean_mr_title(data_frame):
    """
    When a Mr. is under 18 they are an adolescant and should lose their title.
    When a Mr. is under 27 they are 'single'.
    When a Mr. is over 27; flip a coin (53%) whether they are 'single' or
    'married'.

    :param data_frame: a pd.DataFrame
    :return: a pd.DataFrame with mr titles cleaned
    """
    data_frame.loc[
        (data_frame.Title == 'Mr') & (data_frame.Age < 18),
        'Title'
    ] = np.nan
    data_frame.loc[
        (data_frame.Title == 'Mr') & (data_frame.Age < 27),
        'Title'
    ] = 'Single'

    def married_or_single(row):
        if row.Title == 'Mr' and row.Age >= 27:
            row.Title = 'Married' if np.random.rand(1)[0] > 0.53 else 'Single'
        return row.Title

    data_frame.Title = data_frame.apply(married_or_single, axis=1)
    return data_frame


def impute_titles(data_frame):
    data_frame.Title.fillna('None', inplace=True)
    return data_frame
