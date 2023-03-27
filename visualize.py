import os
import streamlit as st
import pandas as pd
import numpy as np

state = st.session_state

def visualize():
    st.title("Visualize")
    st.write("This is the visualize page")

    images_path = st.text_input("Enter the image path")
    masks_path = st.text_input("Enter the masks path")
    state.loaded = st.button("Load")
    if "problematic_images" not in state:
        state.problematic_images = []
    if state.loaded or "images" in state:
        images = list(map(lambda x: os.path.join(images_path, x), os.listdir(images_path)))
        images.sort()
        state.images = images
        st.write(f"Images loaded: {len(images)}")
        masks = list(filter(lambda x: "color" in x, map(lambda x: os.path.join(masks_path, x), os.listdir(masks_path))))
        masks.sort()
        state.maks = masks
        st.write(f"Masks loaded: {len(masks)}")

        scol0, scol1 = st.columns(2)
        with scol0:
            number = st.number_input("Enter the number of images to visualize", min_value=0, max_value=len(images) - 1)
        with scol1:               
            if st.button("Problematic Image"):
                state.problematic_images.append(number)
                st.write("Image added to problematic images")
            
        col0, col1 = st.columns(2)
        with col0:
            st.write("Image", number, ":", images[number])
            st.image(images[number])
        with col1:
            st.write("Mask", number, ":", masks[number])
            st.image(masks[number])
        st.write("Problematic images", state.problematic_images)
        if st.button("Reset"):
            state.problematic_images = []
            st.write("Problematic images reset")

visualize()

"""
problematic_images_007 = [
  22,
  23,
  24,
  32,
  33,
  42,
  43,
  44,
  45,
  55,
  56,
  66,
  67,
  68,
  69,
  80,
  81,
  82,
  91,
  92,
  93,
  94,
  95,
  96,
  104,
  105,
  106,
  108,
  109,
  110,
  116,
  117,
  118,
  120,
  121,
  122,
  123,
  127,
  128,
  129,
  130,
  131,
  132,
  133,
  134,
  135,
  138,
  139,
  140,
  141,
  142,
  143,
  144,
  145,
  146,
  147,
  148,
  149,
  150,
  151,
  152,
  153,
  154,
  155,
  156,
  157,
  158,
  159,
  160,
  161,
  162,
  163,
  164,
  165,
  166,
  167,
  168,
  169,
  170,
  171,
  172,
  173,
  174,
  175,
  176,
  177,
  178,
  179,
  1,
  2,
  6,
  7,
  8,
  9,
  13,
  14,
  15,
  16,
  0
]

problematic_images_005 = [
  2,
  6,
  37,
  45,
  182,
  195,
  198,
  210,
  215,
  218,
  219,
  220,
  223
]

problematic_images_006 = [
  19,
  22,
  23,
  31,
  115,
  127,
  139,
  140
]
"""