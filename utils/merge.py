import numpy as np

def combine(chobj, jlogie_df):
    # to be called by an apply function on a dataframe of change objects as provided by wikiwho
    # depends on jlogie_df as ground truth labels
    boolean = jlogie_df["rev_id"] == chobj["to_rev"]
    token = jlogie_df[boolean]    
    if not token.empty and len(token) == 1:
        which_jlogie = token["token_id"].isin(chobj["ins_tokens"])
        if np.sum(which_jlogie) == 1:
            to_merge = jlogie_df.iloc[which_jlogie.index[0]]
            chobj["nationality"] = to_merge["nationality"]
            chobj["birth_place"] = to_merge["birth_place"]
            chobj["Link"] = to_merge["Link"]
            chobj["Bulk"] = to_merge["Bulk"]
            chobj["token"] = to_merge["token"]
            chobj["action"] = to_merge["action"]
            return chobj
        elif np.sum(which_jlogie) > 1:
            print("more than one row in jlogie_df found!")
            return pd.Series(None)
        elif np.sum(which_jlogie) == 0:
            chobj["nationality"] = None
            chobj["birth_place"] = None
            chobj["Link"] = None
            chobj["Bulk"] = None
            chobj["token"] = None
            chobj["action"] = None
            return chobj
    elif not token.empty and len(token) > 1:
        which_jlogie = token["token_id"].isin(chobj["ins_tokens"]) | token["token_id"].isin(chobj["del_tokens"])
        if np.sum(which_jlogie) == 1:
            to_merge = jlogie_df.iloc[which_jlogie.index[0]]
            chobj["nationality"] = to_merge["nationality"]
            chobj["birth_place"] = to_merge["birth_place"]
            chobj["Link"] = to_merge["Link"]
            chobj["Bulk"] = to_merge["Bulk"]
            chobj["token"] = to_merge["token"]
            chobj["action"] = to_merge["action"]
            return chobj
        elif np.sum(which_jlogie) == 0:
            chobj["nationality"] = None
            chobj["birth_place"] = None
            chobj["Link"] = None
            chobj["Bulk"] = None
            chobj["token"] = None
            chobj["action"] = None
            return chobj
        elif np.sum(which_jlogie) > 1:
            for col in ["nationality", "birth_place", "Link", "Bulk"]:
                if len(token[col].unique()) == 1:
                    chobj[col] = list(token[col])[0]
                else:
                    chobj[col] = 'Y'
                    print("non congruent values found for df['to_rev'] == ", str(chobj["to_rev"]), " and token ids: ", list(token["token_id"]), " in jlogie. Setting Yes to column ", str(col))
            return chobj
        return pd.Series(None)
    else:
        chobj["nationality"] = None
        chobj["birth_place"] = None
        chobj["Link"] = None
        chobj["Bulk"] = None
        chobj["token"] = None
        chobj["action"] = None
        return chobj
