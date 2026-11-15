"""
================================================================================
THERMOPHYSICAL & COMBUSTION MATERIAL DATABASE (118 Engineering & Aerospace Fuels)
================================================================================

Primary Literature & Engineering Handbooks for Material Properties:
--------------------------------------------------------------------------------
1. Tewarson, A. (2002). "Generation of Heat and Chemical Compounds in Fires."
   SFPE Handbook of Fire Protection Engineering (3rd ed.), Section 3, Chapter 4.
   -> Heat of gasification (L_v [kJ/kg]), heat of combustion (dH_c [MJ/kg]),
      and pyrolysis/vaporization temperatures (T_pyro [K]).

2. Quintiere, J. G. (2006). "Fundamentals of Fire Phenomena." John Wiley & Sons.
   Appendix A: Thermophysical properties (rho_s, c_s, k_s) and thermal inertia
   (k*rho*c) for polymers, cellulosics, fabrics, foams, and composites.

3. ASTM D2863 / ISO 4589-2: Standard Test Method for Measuring the Minimum
   Oxygen Concentration to Support Candle-Like Combustion of Plastics (LOI).

4. UL 94: Standard for Safety of Flammability of Plastic Materials for Parts in
   Devices and Appliances (HB, V-2, V-1, V-0 classifications).

5. Incropera, F. P., & DeWitt, D. P. (2007). "Fundamentals of Heat and Mass
   Transfer" (6th ed.). John Wiley & Sons, Appendix A.11.
   -> Total hemispherical infrared surface emissivity (epsilon) for polymers,
      textiles, wood/char, foams, and bare/oxidized metals.

6. Jomaas, G., et al. (2015). Acta Astronautica, 109, 208-216.
   -> NASA Saffire I-VI flight validation materials (SIBAL Standard Fabric,
      PMMA, Nomex 450).
================================================================================
"""

from typing import Dict, List, Any


# Category-level hemispherical IR surface emissivity defaults (Incropera & DeWitt, Table A.11)
CATEGORY_EMISSIVITY_DEFAULTS: Dict[str, float] = {
    "Commodity Polymers": 0.92,              # Non-metallic polymers (0.90-0.95)
    "Engineering Thermosets": 0.90,          # Crosslinked resins / char-forming surfaces
    "High-Performance Thermoplastics": 0.88, # Fluoropolymers & polyaromatics (0.85-0.92)
    "Engineering Elastomers": 0.93,          # Carbon-filled & synthetic rubbers (0.92-0.95)
    "Fabrics & Textiles": 0.88,              # Woven/knit porous fibrous surfaces (0.85-0.90)
    "Foams & Insulation": 0.86,              # Open/closed-cell porous polymer matrices
    "Wood & Bio-materials": 0.90,            # Cellulosic & charred wood surfaces (0.88-0.94)
    "Composites & Laminates": 0.87,          # Resin-rich & carbon/glass fiber laminates
    "Flammable Liquids & Fuels": 0.95,       # Optically thick hydrocarbon/alcohol liquid pools
    "Combustible Metals & Solids": 0.35,     # Bare/oxidized metallic surfaces (0.20-0.45); overridden for waxes/coal
}

# Material-specific overrides where emissivity or microgravity LOI factor differs from category default
MATERIAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Polytetrafluoroethylene (PTFE / Teflon)": {"emissivity": 0.85, "LOI_microg_factor": 0.92},
    "SIBAL Standard Fabric": {"emissivity": 0.89, "LOI_microg_factor": 0.84, "saffire_validated": True},
    "PMMA (Acrylic / Plexiglas)": {"emissivity": 0.92, "LOI_microg_factor": 0.86, "saffire_validated": True},
    "Nomex 450 (Aramid Fabric)": {"emissivity": 0.90, "LOI_microg_factor": 0.85, "saffire_validated": True},
    "Paraffin Candle Wax": {"emissivity": 0.91},
    "Stearic Acid Solid": {"emissivity": 0.91},
    "Asphalt / Bitumen Binder": {"emissivity": 0.94},
    "Coal (Bituminous)": {"emissivity": 0.93},
    "Synthetic Leather (PU / PVC Coated)": {"emissivity": 0.90},
    "Natural Bovine Leather": {"emissivity": 0.90},
}


RAW_MATERIALS_118: List[Dict[str, Any]] = [
    {"name": "PMMA (Acrylic / Plexiglas)", "category": "Commodity Polymers", "rho_s": 1190.0, "c_s": 1420.0, "k_s": 0.19, "T_pyro": 579.0, "L_v_kJkg": 1600.0, "dH_c": 26.2, "LOI": 17.5, "ul94": "HB", "Ea": 130.0, "note": "Rapid burning, clear thermoplastic, low smoke toxicity"},
    {"name": "Low-Density Polyethylene (LDPE)", "category": "Commodity Polymers", "rho_s": 920.0, "c_s": 2300.0, "k_s": 0.33, "T_pyro": 650.0, "L_v_kJkg": 2100.0, "dH_c": 46.5, "LOI": 17.0, "ul94": "HB", "Ea": 220.0, "note": "High heat release rate, heavy dripping during combustion"},
    {"name": "High-Density Polyethylene (HDPE)", "category": "Commodity Polymers", "rho_s": 955.0, "c_s": 2200.0, "k_s": 0.44, "T_pyro": 670.0, "L_v_kJkg": 2200.0, "dH_c": 46.8, "LOI": 17.4, "ul94": "HB", "Ea": 240.0, "note": "Melts and drips burning droplets, high caloric value"},
    {"name": "Polypropylene (PP)", "category": "Commodity Polymers", "rho_s": 905.0, "c_s": 1900.0, "k_s": 0.22, "T_pyro": 620.0, "L_v_kJkg": 2000.0, "dH_c": 45.8, "LOI": 17.5, "ul94": "HB", "Ea": 210.0, "note": "Burns readily with blue flame and yellow tip, olefinic smell"},
    {"name": "General Purpose Polystyrene (GPPS)", "category": "Commodity Polymers", "rho_s": 1050.0, "c_s": 1300.0, "k_s": 0.13, "T_pyro": 630.0, "L_v_kJkg": 1700.0, "dH_c": 41.5, "LOI": 18.0, "ul94": "HB", "Ea": 160.0, "note": "Generates dense black soot and aromatic fumes"},
    {"name": "High Impact Polystyrene (HIPS)", "category": "Commodity Polymers", "rho_s": 1040.0, "c_s": 1400.0, "k_s": 0.14, "T_pyro": 640.0, "L_v_kJkg": 1750.0, "dH_c": 40.8, "LOI": 18.2, "ul94": "HB", "Ea": 165.0, "note": "Rubber-modified PS, high smoke generation"},
    {"name": "Acrylonitrile Butadiene Styrene (ABS)", "category": "Commodity Polymers", "rho_s": 1050.0, "c_s": 1450.0, "k_s": 0.18, "T_pyro": 650.0, "L_v_kJkg": 1850.0, "dH_c": 35.6, "LOI": 18.5, "ul94": "HB", "Ea": 170.0, "note": "Pungent soot, widely used in electronics enclosures"},
    {"name": "Flexible Polyvinyl Chloride (f-PVC)", "category": "Commodity Polymers", "rho_s": 1350.0, "c_s": 1200.0, "k_s": 0.16, "T_pyro": 540.0, "L_v_kJkg": 1400.0, "dH_c": 18.5, "LOI": 24.5, "ul94": "V-2", "Ea": 120.0, "note": "Releases toxic HCl gas and plasticizer vapours upon ignition"},
    {"name": "Rigid Polyvinyl Chloride (r-PVC)", "category": "Commodity Polymers", "rho_s": 1400.0, "c_s": 1050.0, "k_s": 0.15, "T_pyro": 560.0, "L_v_kJkg": 1500.0, "dH_c": 19.8, "LOI": 45.0, "ul94": "V-0", "Ea": 135.0, "note": "Self-extinguishing due to chlorine char formation"},
    {"name": "Polyethylene Terephthalate (PET)", "category": "Commodity Polymers", "rho_s": 1380.0, "c_s": 1200.0, "k_s": 0.24, "T_pyro": 680.0, "L_v_kJkg": 2100.0, "dH_c": 22.5, "LOI": 21.0, "ul94": "HB", "Ea": 190.0, "note": "Melts and chars, commonly used in bottles and synthetic fibers"},
    {"name": "Polycarbonate (PC)", "category": "Commodity Polymers", "rho_s": 1200.0, "c_s": 1200.0, "k_s": 0.2, "T_pyro": 740.0, "L_v_kJkg": 2300.0, "dH_c": 30.8, "LOI": 26.0, "ul94": "V-2", "Ea": 210.0, "note": "Self-extinguishing in thick sections, high smoke yield"},
    {"name": "Polyamide 6 (Nylon 6)", "category": "Commodity Polymers", "rho_s": 1140.0, "c_s": 1600.0, "k_s": 0.25, "T_pyro": 680.0, "L_v_kJkg": 2400.0, "dH_c": 31.2, "LOI": 22.0, "ul94": "V-2", "Ea": 200.0, "note": "Melts and drips, produces burning droplets and HCN traces"},
    {"name": "Polyamide 66 (Nylon 6,6)", "category": "Commodity Polymers", "rho_s": 1150.0, "c_s": 1670.0, "k_s": 0.25, "T_pyro": 700.0, "L_v_kJkg": 2500.0, "dH_c": 31.8, "LOI": 23.0, "ul94": "V-2", "Ea": 205.0, "note": "Higher melting point nylon, dripping burning liquid"},
    {"name": "Polyoxymethylene (POM / Delrin)", "category": "Commodity Polymers", "rho_s": 1410.0, "c_s": 1500.0, "k_s": 0.23, "T_pyro": 600.0, "L_v_kJkg": 2400.0, "dH_c": 16.8, "LOI": 15.0, "ul94": "HB", "Ea": 140.0, "note": "Burns with invisible blue flame, formaldehyde odour"},
    {"name": "Polyphenylene Oxide (PPO)", "category": "Commodity Polymers", "rho_s": 1060.0, "c_s": 1500.0, "k_s": 0.22, "T_pyro": 710.0, "L_v_kJkg": 2200.0, "dH_c": 29.5, "LOI": 28.0, "ul94": "V-1", "Ea": 195.0, "note": "Good inherent flame retardancy, low char yield"},
    {"name": "Styrene Acrylonitrile (SAN)", "category": "Commodity Polymers", "rho_s": 1080.0, "c_s": 1350.0, "k_s": 0.12, "T_pyro": 620.0, "L_v_kJkg": 1750.0, "dH_c": 38.0, "LOI": 18.0, "ul94": "HB", "Ea": 160.0, "note": "Transparent plastic, burns rapidly with black smoke"},
    {"name": "Ethylene Vinyl Acetate (EVA)", "category": "Commodity Polymers", "rho_s": 940.0, "c_s": 2100.0, "k_s": 0.23, "T_pyro": 610.0, "L_v_kJkg": 1900.0, "dH_c": 40.0, "LOI": 18.0, "ul94": "HB", "Ea": 175.0, "note": "Flexible copolymer, releases acetic acid during initial degradation"},
    {"name": "Polybutylene Terephthalate (PBT)", "category": "Commodity Polymers", "rho_s": 1310.0, "c_s": 1300.0, "k_s": 0.21, "T_pyro": 660.0, "L_v_kJkg": 2000.0, "dH_c": 27.0, "LOI": 20.0, "ul94": "HB", "Ea": 185.0, "note": "Rapid crystallization, dripping burn behavior"},
    {"name": "Polymethylpentene (PMP / TPX)", "category": "Commodity Polymers", "rho_s": 830.0, "c_s": 2100.0, "k_s": 0.17, "T_pyro": 640.0, "L_v_kJkg": 2050.0, "dH_c": 44.0, "LOI": 17.1, "ul94": "HB", "Ea": 210.0, "note": "Ultra-lightweight polyolefin, clear burning flame"},
    {"name": "Polybutylene (PB-1)", "category": "Commodity Polymers", "rho_s": 915.0, "c_s": 1950.0, "k_s": 0.2, "T_pyro": 630.0, "L_v_kJkg": 2000.0, "dH_c": 45.0, "LOI": 17.2, "ul94": "HB", "Ea": 200.0, "note": "Flexible piping resin, burns with paraffinic odour"},
    {"name": "Epoxy Resin (Unfilled)", "category": "Engineering Thermosets", "rho_s": 1200.0, "c_s": 1250.0, "k_s": 0.21, "T_pyro": 630.0, "L_v_kJkg": 1800.0, "dH_c": 29.0, "LOI": 19.8, "ul94": "HB", "Ea": 150.0, "note": "Thermoset, forms carbonaceous char, intense black smoke"},
    {"name": "Unsaturated Polyester Resin (UPR)", "category": "Engineering Thermosets", "rho_s": 1280.0, "c_s": 1300.0, "k_s": 0.19, "T_pyro": 610.0, "L_v_kJkg": 1750.0, "dH_c": 26.5, "LOI": 18.5, "ul94": "HB", "Ea": 140.0, "note": "Styrene-crosslinked matrix, rapid heat release rate"},
    {"name": "Phenolic Resin (Bakelite)", "category": "Engineering Thermosets", "rho_s": 1350.0, "c_s": 1400.0, "k_s": 0.25, "T_pyro": 770.0, "L_v_kJkg": 2800.0, "dH_c": 26.0, "LOI": 35.0, "ul94": "V-0", "Ea": 220.0, "note": "High char yield (>50%), excellent heat resistance"},
    {"name": "Melamine Formaldehyde (MF)", "category": "Engineering Thermosets", "rho_s": 1500.0, "c_s": 1450.0, "k_s": 0.32, "T_pyro": 650.0, "L_v_kJkg": 2500.0, "dH_c": 18.0, "LOI": 32.0, "ul94": "V-0", "Ea": 190.0, "note": "Nitrogen-rich resin, acts as inherent flame retardant"},
    {"name": "Urea Formaldehyde (UF)", "category": "Engineering Thermosets", "rho_s": 1450.0, "c_s": 1400.0, "k_s": 0.28, "T_pyro": 590.0, "L_v_kJkg": 2200.0, "dH_c": 16.5, "LOI": 30.0, "ul94": "V-1", "Ea": 165.0, "note": "Adhesive resin, releases ammonia and formaldehyde gas"},
    {"name": "Polyimide (PI / Kapton)", "category": "Engineering Thermosets", "rho_s": 1420.0, "c_s": 1100.0, "k_s": 0.12, "T_pyro": 820.0, "L_v_kJkg": 3200.0, "dH_c": 31.0, "LOI": 37.0, "ul94": "V-0", "Ea": 260.0, "note": "Aerospace film polymer, extremely fire resistant"},
    {"name": "Polyether Ether Ketone (PEEK)", "category": "High-Performance Thermoplastics", "rho_s": 1320.0, "c_s": 1340.0, "k_s": 0.25, "T_pyro": 850.0, "L_v_kJkg": 3400.0, "dH_c": 31.5, "LOI": 35.0, "ul94": "V-0", "Ea": 270.0, "note": "High temperature performance polymer, minimal smoke"},
    {"name": "Polyethersulfone (PES)", "category": "High-Performance Thermoplastics", "rho_s": 1370.0, "c_s": 1120.0, "k_s": 0.18, "T_pyro": 790.0, "L_v_kJkg": 2900.0, "dH_c": 26.8, "LOI": 38.0, "ul94": "V-0", "Ea": 240.0, "note": "Sulfone linkage provides flame retardancy and char stability"},
    {"name": "Polyphenylene Sulfide (PPS)", "category": "High-Performance Thermoplastics", "rho_s": 1350.0, "c_s": 1100.0, "k_s": 0.29, "T_pyro": 780.0, "L_v_kJkg": 3000.0, "dH_c": 28.0, "LOI": 40.0, "ul94": "V-0", "Ea": 250.0, "note": "Inherent FR resin, releases SO2 traces during forced flaming"},
    {"name": "Polyamide-imide (PAI / Torlon)", "category": "Engineering Thermosets", "rho_s": 1420.0, "c_s": 1050.0, "k_s": 0.26, "T_pyro": 810.0, "L_v_kJkg": 3300.0, "dH_c": 29.5, "LOI": 45.0, "ul94": "V-0", "Ea": 265.0, "note": "Exceptional char density and extreme temperature stability"},
    {"name": "Polyvinylidene Fluoride (PVDF)", "category": "High-Performance Thermoplastics", "rho_s": 1780.0, "c_s": 1300.0, "k_s": 0.19, "T_pyro": 690.0, "L_v_kJkg": 2300.0, "dH_c": 13.5, "LOI": 44.0, "ul94": "V-0", "Ea": 210.0, "note": "Fluoropolymer, low flame spread and low heat release"},
    {"name": "Polytetrafluoroethylene (PTFE / Teflon)", "category": "High-Performance Thermoplastics", "rho_s": 2150.0, "c_s": 1000.0, "k_s": 0.25, "T_pyro": 830.0, "L_v_kJkg": 1700.0, "dH_c": 5.1, "LOI": 95.0, "ul94": "V-0", "Ea": 350.0, "note": "Non-flammable in air, extreme LOI, pyrolyzes toxic HF/PFIB"},
    {"name": "Fluorinated Ethylene Propylene (FEP)", "category": "High-Performance Thermoplastics", "rho_s": 2130.0, "c_s": 1100.0, "k_s": 0.2, "T_pyro": 800.0, "L_v_kJkg": 1800.0, "dH_c": 5.3, "LOI": 95.0, "ul94": "V-0", "Ea": 330.0, "note": "Melt-processable fluoropolymer, practically non-combustible"},
    {"name": "Perfluoroalkoxy Alkane (PFA)", "category": "High-Performance Thermoplastics", "rho_s": 2150.0, "c_s": 1050.0, "k_s": 0.19, "T_pyro": 810.0, "L_v_kJkg": 1850.0, "dH_c": 5.2, "LOI": 95.0, "ul94": "V-0", "Ea": 340.0, "note": "High purity fluoropolymer, extreme chemical and fire resistance"},
    {"name": "Chlorinated Polyvinyl Chloride (CPVC)", "category": "High-Performance Thermoplastics", "rho_s": 1550.0, "c_s": 1000.0, "k_s": 0.14, "T_pyro": 700.0, "L_v_kJkg": 2000.0, "dH_c": 17.0, "LOI": 60.0, "ul94": "V-0", "Ea": 180.0, "note": "Higher chlorine content than PVC, used in fire sprinkler pipe"},
    {"name": "Silicone Rubber (Polydimethylsiloxane)", "category": "Engineering Elastomers", "rho_s": 1200.0, "c_s": 1250.0, "k_s": 0.2, "T_pyro": 670.0, "L_v_kJkg": 2100.0, "dH_c": 18.0, "LOI": 22.0, "ul94": "HB", "Ea": 180.0, "note": "Forms white silica ash residue, low toxic gas emissions"},
    {"name": "Nitrile Butadiene Rubber (NBR)", "category": "Engineering Elastomers", "rho_s": 1000.0, "c_s": 1900.0, "k_s": 0.25, "T_pyro": 630.0, "L_v_kJkg": 2200.0, "dH_c": 38.0, "LOI": 19.0, "ul94": "HB", "Ea": 160.0, "note": "Synthetic rubber, heavy black soot generation"},
    {"name": "Ethylene Propylene Diene Rubber (EPDM)", "category": "Engineering Elastomers", "rho_s": 980.0, "c_s": 2100.0, "k_s": 0.22, "T_pyro": 640.0, "L_v_kJkg": 2300.0, "dH_c": 42.0, "LOI": 18.0, "ul94": "HB", "Ea": 190.0, "note": "Olefin elastomer, burns readily with high heat release"},
    {"name": "Styrene-Butadiene Rubber (SBR)", "category": "Engineering Elastomers", "rho_s": 940.0, "c_s": 1800.0, "k_s": 0.19, "T_pyro": 620.0, "L_v_kJkg": 2100.0, "dH_c": 41.0, "LOI": 18.5, "ul94": "HB", "Ea": 170.0, "note": "Tire rubber polymer, dense toxic smoke and oil runoff"},
    {"name": "Natural Rubber (Isoprene)", "category": "Engineering Elastomers", "rho_s": 920.0, "c_s": 1880.0, "k_s": 0.13, "T_pyro": 600.0, "L_v_kJkg": 2000.0, "dH_c": 44.0, "LOI": 17.5, "ul94": "HB", "Ea": 150.0, "note": "Burns rapidly with strong rubber smoke and tacky liquid residue"},
    {"name": "Cotton Fabric (Untreated)", "category": "Fabrics & Textiles", "rho_s": 1540.0, "c_s": 1300.0, "k_s": 0.06, "T_pyro": 630.0, "L_v_kJkg": 2400.0, "dH_c": 17.2, "LOI": 16.5, "ul94": "HB", "Ea": 150.0, "note": "Rapid surface flame propagation, glowing smoldering char"},
    {"name": "Cotton Jersey Fabric", "category": "Fabrics & Textiles", "rho_s": 1540.0, "c_s": 1300.0, "k_s": 0.06, "T_pyro": 630.0, "L_v_kJkg": 2400.0, "dH_c": 17.2, "LOI": 16.5, "ul94": "HB", "Ea": 150.0, "note": "Knit apparel fabric, rapid upward flame spread"},
    {"name": "Wool Fabric", "category": "Fabrics & Textiles", "rho_s": 1310.0, "c_s": 1360.0, "k_s": 0.04, "T_pyro": 570.0, "L_v_kJkg": 2100.0, "dH_c": 20.5, "LOI": 25.2, "ul94": "V-2", "Ea": 140.0, "note": "Protein fiber, self-extinguishing in horizontal orientation"},
    {"name": "Silk Fabric", "category": "Fabrics & Textiles", "rho_s": 1300.0, "c_s": 1380.0, "k_s": 0.05, "T_pyro": 580.0, "L_v_kJkg": 2150.0, "dH_c": 19.8, "LOI": 23.5, "ul94": "HB", "Ea": 145.0, "note": "Natural protein fiber, burns slowly with burning hair odour"},
    {"name": "Viscose Rayon Fabric", "category": "Fabrics & Textiles", "rho_s": 1500.0, "c_s": 1350.0, "k_s": 0.06, "T_pyro": 610.0, "L_v_kJkg": 2350.0, "dH_c": 17.5, "LOI": 17.0, "ul94": "HB", "Ea": 145.0, "note": "Regenerated cellulose, flammability identical to raw cotton"},
    {"name": "Linen (Flax Fabric)", "category": "Fabrics & Textiles", "rho_s": 1500.0, "c_s": 1320.0, "k_s": 0.06, "T_pyro": 620.0, "L_v_kJkg": 2400.0, "dH_c": 17.0, "LOI": 16.8, "ul94": "HB", "Ea": 150.0, "note": "Cellulosic fiber, rapid ignitability and smoldering propensity"},
    {"name": "Polyester Fabric (PET Textile)", "category": "Fabrics & Textiles", "rho_s": 1380.0, "c_s": 1200.0, "k_s": 0.05, "T_pyro": 680.0, "L_v_kJkg": 2100.0, "dH_c": 22.0, "LOI": 21.0, "ul94": "V-2", "Ea": 190.0, "note": "Melts away from flame, can cause severe skin melt burns"},
    {"name": "Nylon 6 Fabric", "category": "Fabrics & Textiles", "rho_s": 1140.0, "c_s": 1600.0, "k_s": 0.06, "T_pyro": 680.0, "L_v_kJkg": 2400.0, "dH_c": 31.0, "LOI": 21.5, "ul94": "V-2", "Ea": 200.0, "note": "Synthetic fiber, melts and drips burning beads"},
    {"name": "Acrylic Textile Fabric", "category": "Fabrics & Textiles", "rho_s": 1180.0, "c_s": 1400.0, "k_s": 0.05, "T_pyro": 560.0, "L_v_kJkg": 1900.0, "dH_c": 31.5, "LOI": 18.2, "ul94": "HB", "Ea": 135.0, "note": "Polyacrylonitrile fiber, burns vigorously with hot melt drip"},
    {"name": "Nomex 450 (Aramid Fabric)", "category": "Fabrics & Textiles", "rho_s": 1380.0, "c_s": 1200.0, "k_s": 0.13, "T_pyro": 780.0, "L_v_kJkg": 3100.0, "dH_c": 27.0, "LOI": 28.5, "ul94": "V-0", "Ea": 220.0, "note": "Meta-aramid fabric, protective apparel standard, non-melting char"},
    {"name": "Kevlar 29 (Para-Aramid Fabric)", "category": "Fabrics & Textiles", "rho_s": 1440.0, "c_s": 1420.0, "k_s": 0.04, "T_pyro": 830.0, "L_v_kJkg": 3500.0, "dH_c": 28.5, "LOI": 29.0, "ul94": "V-0", "Ea": 250.0, "note": "High strength aramid fiber, chars without melting"},
    {"name": "SIBAL Standard Fabric", "category": "Fabrics & Textiles", "rho_s": 1800.0, "c_s": 1250.0, "k_s": 0.28, "T_pyro": 620.0, "L_v_kJkg": 2200.0, "dH_c": 18.5, "LOI": 16.0, "ul94": "HB", "Ea": 160.0, "note": "Cotton-fiberglass blend flight test fuel used in NASA Saffire"},
    {"name": "Cotton-Polyester Blend (50/50)", "category": "Fabrics & Textiles", "rho_s": 1450.0, "c_s": 1250.0, "k_s": 0.05, "T_pyro": 640.0, "L_v_kJkg": 2250.0, "dH_c": 19.5, "LOI": 18.0, "ul94": "HB", "Ea": 165.0, "note": "Scaffolding burning effect: cotton holds skeleton while PET drips"},
    {"name": "Modacrylic Fabric", "category": "Fabrics & Textiles", "rho_s": 1350.0, "c_s": 1300.0, "k_s": 0.05, "T_pyro": 590.0, "L_v_kJkg": 2000.0, "dH_c": 24.0, "LOI": 29.0, "ul94": "V-0", "Ea": 170.0, "note": "Inherent FR copolymer containing vinyl chloride/vinylidene chloride"},
    {"name": "Jute Burlap Fabric", "category": "Fabrics & Textiles", "rho_s": 1450.0, "c_s": 1300.0, "k_s": 0.07, "T_pyro": 580.0, "L_v_kJkg": 2200.0, "dH_c": 16.5, "LOI": 16.0, "ul94": "HB", "Ea": 140.0, "note": "Coarse lignocellulosic fiber, highly flammable loose weave"},
    {"name": "Hemp Fabric", "category": "Fabrics & Textiles", "rho_s": 1480.0, "c_s": 1320.0, "k_s": 0.06, "T_pyro": 600.0, "L_v_kJkg": 2300.0, "dH_c": 16.8, "LOI": 16.5, "ul94": "HB", "Ea": 145.0, "note": "Bast plant fiber, burns rapidly with cellulosic ash"},
    {"name": "Polypropylene Non-Woven Textile", "category": "Fabrics & Textiles", "rho_s": 900.0, "c_s": 1900.0, "k_s": 0.04, "T_pyro": 620.0, "L_v_kJkg": 2000.0, "dH_c": 45.5, "LOI": 17.5, "ul94": "HB", "Ea": 210.0, "note": "Surgical mask material, melts rapidly and fuels flame spread"},
    {"name": "FR-Treated Cotton (Proban/Pyrovatex)", "category": "Fabrics & Textiles", "rho_s": 1560.0, "c_s": 1280.0, "k_s": 0.07, "T_pyro": 550.0, "L_v_kJkg": 1800.0, "dH_c": 14.5, "LOI": 28.0, "ul94": "V-0", "Ea": 130.0, "note": "Phosphorus chemical treatment promotes heavy charring at lower T"},
    {"name": "Flexible Polyurethane Foam (FPUR)", "category": "Foams & Insulation", "rho_s": 30.0, "c_s": 1400.0, "k_s": 0.04, "T_pyro": 560.0, "L_v_kJkg": 1200.0, "dH_c": 27.2, "LOI": 16.5, "ul94": "HB", "Ea": 130.0, "note": "Extreme fire hazard in upholstery, flashover driver, pool fire formation"},
    {"name": "Rigid Polyurethane Foam (RPUR)", "category": "Foams & Insulation", "rho_s": 40.0, "c_s": 1350.0, "k_s": 0.02, "T_pyro": 580.0, "L_v_kJkg": 1300.0, "dH_c": 25.0, "LOI": 18.0, "ul94": "HB", "Ea": 140.0, "note": "Building insulation foam, intense toxic cyanic smoke"},
    {"name": "Expanded Polystyrene Foam (EPS)", "category": "Foams & Insulation", "rho_s": 20.0, "c_s": 1200.0, "k_s": 0.03, "T_pyro": 620.0, "L_v_kJkg": 1100.0, "dH_c": 40.0, "LOI": 17.5, "ul94": "HB", "Ea": 160.0, "note": "Melts rapidly at 100°C into liquid styrenic fuel pool"},
    {"name": "Extruded Polystyrene Foam (XPS)", "category": "Foams & Insulation", "rho_s": 35.0, "c_s": 1250.0, "k_s": 0.03, "T_pyro": 630.0, "L_v_kJkg": 1150.0, "dH_c": 40.5, "LOI": 18.0, "ul94": "HB", "Ea": 165.0, "note": "Dense insulation foam, severe melt flow and black smoke"},
    {"name": "Polyisocyanurate Foam (PIR)", "category": "Foams & Insulation", "rho_s": 32.0, "c_s": 1300.0, "k_s": 0.02, "T_pyro": 620.0, "L_v_kJkg": 1600.0, "dH_c": 24.0, "LOI": 23.0, "ul94": "V-1", "Ea": 160.0, "note": "Improved char structure compared to PUR foam"},
    {"name": "Polyethylene Foam (PE Foam)", "category": "Foams & Insulation", "rho_s": 35.0, "c_s": 2100.0, "k_s": 0.04, "T_pyro": 650.0, "L_v_kJkg": 1500.0, "dH_c": 46.0, "LOI": 17.0, "ul94": "HB", "Ea": 220.0, "note": "Packaging foam, burns with dripping liquid paraffin droplets"},
    {"name": "Melamine Foam (Basotect)", "category": "Foams & Insulation", "rho_s": 10.0, "c_s": 1200.0, "k_s": 0.04, "T_pyro": 650.0, "L_v_kJkg": 2500.0, "dH_c": 17.5, "LOI": 32.0, "ul94": "V-0", "Ea": 190.0, "note": "Acoustic foam, inherently fire resistant without additives"},
    {"name": "Phenolic Foam Insulation", "category": "Foams & Insulation", "rho_s": 45.0, "c_s": 1300.0, "k_s": 0.02, "T_pyro": 750.0, "L_v_kJkg": 2600.0, "dH_c": 25.0, "LOI": 38.0, "ul94": "V-0", "Ea": 210.0, "note": "Resistant roof/pipe insulation, minimal smoke and charring"},
    {"name": "Latex Rubber Foam", "category": "Foams & Insulation", "rho_s": 70.0, "c_s": 1800.0, "k_s": 0.04, "T_pyro": 590.0, "L_v_kJkg": 1400.0, "dH_c": 42.0, "LOI": 17.0, "ul94": "HB", "Ea": 145.0, "note": "Mattress foam, ignites easily, smolders with high heat output"},
    {"name": "Polypropylene Foam (EPP)", "category": "Foams & Insulation", "rho_s": 45.0, "c_s": 1850.0, "k_s": 0.04, "T_pyro": 620.0, "L_v_kJkg": 1500.0, "dH_c": 45.0, "LOI": 17.5, "ul94": "HB", "Ea": 200.0, "note": "Automotive foam bumper material, high heat release rate"},
    {"name": "PVC Foam Board (Sintra / Forex)", "category": "Foams & Insulation", "rho_s": 500.0, "c_s": 1100.0, "k_s": 0.07, "T_pyro": 550.0, "L_v_kJkg": 1600.0, "dH_c": 19.0, "LOI": 32.0, "ul94": "V-0", "Ea": 130.0, "note": "Signage board, self-extinguishing charring rigid foam"},
    {"name": "Syntactic Foam (Epoxy-Glass Microballoons)", "category": "Foams & Insulation", "rho_s": 600.0, "c_s": 1200.0, "k_s": 0.12, "T_pyro": 650.0, "L_v_kJkg": 2000.0, "dH_c": 22.0, "LOI": 24.0, "ul94": "V-1", "Ea": 160.0, "note": "Subsea buoyant foam, glass microballoons retard fire spread"},
    {"name": "Polyimide Foam (Solimide)", "category": "Foams & Insulation", "rho_s": 8.0, "c_s": 1100.0, "k_s": 0.04, "T_pyro": 800.0, "L_v_kJkg": 3200.0, "dH_c": 30.0, "LOI": 42.0, "ul94": "V-0", "Ea": 250.0, "note": "Ultra-lightweight aerospace insulation, highly fire resistant"},
    {"name": "Cellular Rubber Insulation (Armaflex / NBR-PVC)", "category": "Foams & Insulation", "rho_s": 60.0, "c_s": 1500.0, "k_s": 0.04, "T_pyro": 580.0, "L_v_kJkg": 1600.0, "dH_c": 26.0, "LOI": 32.0, "ul94": "V-0", "Ea": 140.0, "note": "Elastomeric pipe lagging, fire retardant additives"},
    {"name": "Wood Fiber Insulation Board", "category": "Foams & Insulation", "rho_s": 160.0, "c_s": 1600.0, "k_s": 0.04, "T_pyro": 570.0, "L_v_kJkg": 1800.0, "dH_c": 18.0, "LOI": 18.0, "ul94": "HB", "Ea": 140.0, "note": "Bio-based insulation, susceptible to smoldering combustion"},
    {"name": "Douglas Fir Timber", "category": "Wood & Bio-materials", "rho_s": 530.0, "c_s": 1300.0, "k_s": 0.12, "T_pyro": 593.0, "L_v_kJkg": 1800.0, "dH_c": 19.5, "LOI": 18.0, "ul94": "HB", "Ea": 150.0, "note": "Structural softwood timber, steady charring rate 0.6 mm/min"},
    {"name": "Southern Yellow Pine", "category": "Wood & Bio-materials", "rho_s": 590.0, "c_s": 1350.0, "k_s": 0.14, "T_pyro": 590.0, "L_v_kJkg": 1750.0, "dH_c": 20.0, "LOI": 18.2, "ul94": "HB", "Ea": 155.0, "note": "Resinous softwood, pitch content accelerates flaming"},
    {"name": "White Pine Timber", "category": "Wood & Bio-materials", "rho_s": 420.0, "c_s": 1300.0, "k_s": 0.11, "T_pyro": 580.0, "L_v_kJkg": 1850.0, "dH_c": 19.0, "LOI": 17.8, "ul94": "HB", "Ea": 145.0, "note": "Low density softwood, fast ignition and flame spread"},
    {"name": "Red Oak Lumber", "category": "Wood & Bio-materials", "rho_s": 680.0, "c_s": 1250.0, "k_s": 0.17, "T_pyro": 610.0, "L_v_kJkg": 1900.0, "dH_c": 18.8, "LOI": 19.0, "ul94": "HB", "Ea": 160.0, "note": "Dense hardwood standard fuel, char rate 0.5 mm/min"},
    {"name": "Sugar Maple Lumber", "category": "Wood & Bio-materials", "rho_s": 720.0, "c_s": 1250.0, "k_s": 0.18, "T_pyro": 615.0, "L_v_kJkg": 1950.0, "dH_c": 18.5, "LOI": 19.2, "ul94": "HB", "Ea": 165.0, "note": "Dense hardwood, structural integrity retained behind char layer"},
    {"name": "Balsa Wood", "category": "Wood & Bio-materials", "rho_s": 160.0, "c_s": 1300.0, "k_s": 0.05, "T_pyro": 550.0, "L_v_kJkg": 1600.0, "dH_c": 18.0, "LOI": 16.0, "ul94": "HB", "Ea": 130.0, "note": "Ultra-light wood, rapid flame velocity and total consumption"},
    {"name": "Birch Plywood", "category": "Wood & Bio-materials", "rho_s": 650.0, "c_s": 1350.0, "k_s": 0.13, "T_pyro": 590.0, "L_v_kJkg": 1850.0, "dH_c": 18.2, "LOI": 18.5, "ul94": "HB", "Ea": 150.0, "note": "Laminated wood sheet, adhesive vaporizes and delaminates fire side"},
    {"name": "Particle Board (Chipboard)", "category": "Wood & Bio-materials", "rho_s": 680.0, "c_s": 1400.0, "k_s": 0.13, "T_pyro": 580.0, "L_v_kJkg": 1750.0, "dH_c": 17.5, "LOI": 18.0, "ul94": "HB", "Ea": 145.0, "note": "Resin-bound wood chips, resin binder contributes volatile fuel"},
    {"name": "Medium Density Fiberboard (MDF)", "category": "Wood & Bio-materials", "rho_s": 750.0, "c_s": 1420.0, "k_s": 0.14, "T_pyro": 585.0, "L_v_kJkg": 1800.0, "dH_c": 17.8, "LOI": 18.2, "ul94": "HB", "Ea": 150.0, "note": "Homogeneous wood fiber matrix, sustained smoldering propensity"},
    {"name": "Hardboard (High-Density Fiberboard)", "category": "Wood & Bio-materials", "rho_s": 950.0, "c_s": 1380.0, "k_s": 0.16, "T_pyro": 595.0, "L_v_kJkg": 1850.0, "dH_c": 18.0, "LOI": 18.8, "ul94": "HB", "Ea": 155.0, "note": "Compressed cellulosic panel, slow char depth growth"},
    {"name": "Bamboo Plywood", "category": "Wood & Bio-materials", "rho_s": 700.0, "c_s": 1320.0, "k_s": 0.17, "T_pyro": 600.0, "L_v_kJkg": 1900.0, "dH_c": 19.2, "LOI": 18.5, "ul94": "HB", "Ea": 155.0, "note": "Monocot grass fiber composite, high initial peak heat release"},
    {"name": "Kimwipes Cellulosic Tissues", "category": "Wood & Bio-materials", "rho_s": 350.0, "c_s": 1250.0, "k_s": 0.05, "T_pyro": 560.0, "L_v_kJkg": 2100.0, "dH_c": 16.5, "LOI": 16.0, "ul94": "HB", "Ea": 130.0, "note": "Thermally thin tissue fuel, baseline flammability calibration target"},
    {"name": "Ashless Filter Paper (Whatman 40)", "category": "Wood & Bio-materials", "rho_s": 450.0, "c_s": 1250.0, "k_s": 0.06, "T_pyro": 570.0, "L_v_kJkg": 2200.0, "dH_c": 16.8, "LOI": 16.2, "ul94": "HB", "Ea": 135.0, "note": "Pure cellulose sheet, standard microgravity ignition material"},
    {"name": "Corrugated Cardboard", "category": "Wood & Bio-materials", "rho_s": 150.0, "c_s": 1300.0, "k_s": 0.06, "T_pyro": 570.0, "L_v_kJkg": 1700.0, "dH_c": 16.0, "LOI": 16.5, "ul94": "HB", "Ea": 135.0, "note": "Air fluting enhances internal flame spread and rapid burn-through"},
    {"name": "Nitrocellulose (Pyroxylin)", "category": "Wood & Bio-materials", "rho_s": 1650.0, "c_s": 1200.0, "k_s": 0.23, "T_pyro": 440.0, "L_v_kJkg": 900.0, "dH_c": 10.5, "LOI": 11.0, "ul94": "HB", "Ea": 85.0, "note": "Energetic polymer, self-oxidizing decomposition, extreme hazard"},
    {"name": "CFRP (Carbon Fiber Epoxy)", "category": "Composites & Laminates", "rho_s": 1550.0, "c_s": 1100.0, "k_s": 4.5, "T_pyro": 650.0, "L_v_kJkg": 2100.0, "dH_c": 28.0, "LOI": 24.0, "ul94": "V-1", "Ea": 160.0, "note": "Structural aerospace composite, epoxy burns leaving bare carbon fiber"},
    {"name": "GFRP (Glass Fiber Polyester)", "category": "Composites & Laminates", "rho_s": 1800.0, "c_s": 1150.0, "k_s": 0.35, "T_pyro": 620.0, "L_v_kJkg": 1900.0, "dH_c": 20.0, "LOI": 21.0, "ul94": "HB", "Ea": 145.0, "note": "Marine fiberglass, glass mat remains behind as thermal barrier"},
    {"name": "FR-4 PCB Laminate (Epoxy-Glass FR)", "category": "Composites & Laminates", "rho_s": 1850.0, "c_s": 1100.0, "k_s": 0.3, "T_pyro": 580.0, "L_v_kJkg": 2200.0, "dH_c": 15.0, "LOI": 35.0, "ul94": "V-0", "Ea": 170.0, "note": "Brominated FR epoxy electronics substrate standard"},
    {"name": "Phenolic-Cotton Laminate (Bakelite C)", "category": "Composites & Laminates", "rho_s": 1380.0, "c_s": 1350.0, "k_s": 0.28, "T_pyro": 720.0, "L_v_kJkg": 2600.0, "dH_c": 22.0, "LOI": 32.0, "ul94": "V-0", "Ea": 210.0, "note": "Industrial electrical laminate, excellent char and flame resistance"},
    {"name": "Wood-Plastic Composite (WPC - PE/Wood)", "category": "Composites & Laminates", "rho_s": 1100.0, "c_s": 1600.0, "k_s": 0.25, "T_pyro": 630.0, "L_v_kJkg": 1950.0, "dH_c": 32.0, "LOI": 17.5, "ul94": "HB", "Ea": 180.0, "note": "Decking composite, combines wood flame spread and PE liquid drip"},
    {"name": "Aramid-Epoxy Composite (Kevlar/Epoxy)", "category": "Composites & Laminates", "rho_s": 1380.0, "c_s": 1250.0, "k_s": 0.22, "T_pyro": 680.0, "L_v_kJkg": 2400.0, "dH_c": 28.5, "LOI": 28.0, "ul94": "V-1", "Ea": 190.0, "note": "Ballistic structure composite, high char retention"},
    {"name": "CFRP (Carbon Fiber PEEK Matrix)", "category": "Composites & Laminates", "rho_s": 1580.0, "c_s": 1200.0, "k_s": 5.2, "T_pyro": 850.0, "L_v_kJkg": 3400.0, "dH_c": 30.0, "LOI": 42.0, "ul94": "V-0", "Ea": 270.0, "note": "High performance thermoplastic composite, minimal heat release"},
    {"name": "Glass-Phenolic Structural Laminate", "category": "Composites & Laminates", "rho_s": 1750.0, "c_s": 1200.0, "k_s": 0.32, "T_pyro": 750.0, "L_v_kJkg": 2800.0, "dH_c": 16.0, "LOI": 45.0, "ul94": "V-0", "Ea": 230.0, "note": "Aircraft interior panel composite, low smoke and toxicity"},
    {"name": "Honeycomb Sandwich (Nomex Core / Epoxy Skin)", "category": "Composites & Laminates", "rho_s": 120.0, "c_s": 1200.0, "k_s": 0.08, "T_pyro": 650.0, "L_v_kJkg": 2200.0, "dH_c": 25.0, "LOI": 28.0, "ul94": "V-0", "Ea": 170.0, "note": "Aircraft cabin floorboard, core air void accelerates internal heating"},
    {"name": "Silicone-Glass Cloth Laminate", "category": "Composites & Laminates", "rho_s": 1820.0, "c_s": 1150.0, "k_s": 0.28, "T_pyro": 700.0, "L_v_kJkg": 2300.0, "dH_c": 14.0, "LOI": 38.0, "ul94": "V-0", "Ea": 200.0, "note": "Flexible high-temp electrical insulation barrier"},
    {"name": "Methanol Liquid Fuel (Pool)", "category": "Flammable Liquids & Fuels", "rho_s": 792.0, "c_s": 2530.0, "k_s": 0.2, "T_pyro": 338.0, "L_v_kJkg": 1100.0, "dH_c": 22.7, "LOI": 11.0, "ul94": "HB", "Ea": 60.0, "note": "Invisible pale blue flame, severe flash fire hazard"},
    {"name": "Ethanol Liquid Fuel (Pool)", "category": "Flammable Liquids & Fuels", "rho_s": 789.0, "c_s": 2440.0, "k_s": 0.17, "T_pyro": 351.0, "L_v_kJkg": 920.0, "dH_c": 29.7, "LOI": 13.0, "ul94": "HB", "Ea": 65.0, "note": "Clean burning alcohol pool fire, low soot radiation"},
    {"name": "Isopropanol (IPA Pool)", "category": "Flammable Liquids & Fuels", "rho_s": 786.0, "c_s": 2600.0, "k_s": 0.14, "T_pyro": 355.0, "L_v_kJkg": 750.0, "dH_c": 33.1, "LOI": 13.5, "ul94": "HB", "Ea": 70.0, "note": "Common solvent fuel, luminous yellow tipped flame"},
    {"name": "n-Heptane Liquid Fuel", "category": "Flammable Liquids & Fuels", "rho_s": 684.0, "c_s": 2240.0, "k_s": 0.12, "T_pyro": 371.0, "L_v_kJkg": 360.0, "dH_c": 48.1, "LOI": 12.0, "ul94": "HB", "Ea": 75.0, "note": "Standard hydrocarbon pool fire reference fuel"},
    {"name": "Toluene Liquid Fuel", "category": "Flammable Liquids & Fuels", "rho_s": 867.0, "c_s": 1700.0, "k_s": 0.13, "T_pyro": 384.0, "L_v_kJkg": 380.0, "dH_c": 42.4, "LOI": 13.0, "ul94": "HB", "Ea": 80.0, "note": "Aromatic solvent fuel, intense black soot producing fire"},
    {"name": "Acetone Solvent", "category": "Flammable Liquids & Fuels", "rho_s": 784.0, "c_s": 2150.0, "k_s": 0.16, "T_pyro": 329.0, "L_v_kJkg": 520.0, "dH_c": 30.8, "LOI": 13.0, "ul94": "HB", "Ea": 55.0, "note": "Highly volatile solvent, vapor flash fire risk"},
    {"name": "Kerosene / Jet-A Aviation Fuel", "category": "Flammable Liquids & Fuels", "rho_s": 800.0, "c_s": 2010.0, "k_s": 0.15, "T_pyro": 450.0, "L_v_kJkg": 300.0, "dH_c": 43.5, "LOI": 12.5, "ul94": "HB", "Ea": 90.0, "note": "Aviation turbine fuel, massive thermal radiation in crash fires"},
    {"name": "Diesel Fuel / Heating Oil", "category": "Flammable Liquids & Fuels", "rho_s": 835.0, "c_s": 2000.0, "k_s": 0.14, "T_pyro": 480.0, "L_v_kJkg": 280.0, "dH_c": 44.8, "LOI": 13.0, "ul94": "HB", "Ea": 95.0, "note": "Combustible liquid class II, heavy soot and thermal flux"},
    {"name": "Transformer Mineral Oil", "category": "Flammable Liquids & Fuels", "rho_s": 880.0, "c_s": 1900.0, "k_s": 0.13, "T_pyro": 520.0, "L_v_kJkg": 250.0, "dH_c": 46.0, "LOI": 14.0, "ul94": "HB", "Ea": 110.0, "note": "High flash point dielectric oil, violent pool fires upon arc fault"},
    {"name": "Hydraulic Fluid (Mineral Base)", "category": "Flammable Liquids & Fuels", "rho_s": 870.0, "c_s": 1950.0, "k_s": 0.13, "T_pyro": 510.0, "L_v_kJkg": 260.0, "dH_c": 45.2, "LOI": 14.0, "ul94": "HB", "Ea": 105.0, "note": "High pressure mist atomization creates explosive aerosol cloud"},
    {"name": "Magnesium Solid Metal (Ribbon/Powder)", "category": "Combustible Metals & Solids", "rho_s": 1740.0, "c_s": 1020.0, "k_s": 156.0, "T_pyro": 923.0, "L_v_kJkg": 5200.0, "dH_c": 24.8, "LOI": 8.0, "ul94": "HB", "Ea": 150.0, "note": "Class D metal fire, brilliant white ultraviolet flame at 3100°C"},
    {"name": "Titanium Metal Powder", "category": "Combustible Metals & Solids", "rho_s": 4500.0, "c_s": 520.0, "k_s": 21.9, "T_pyro": 1941.0, "L_v_kJkg": 8800.0, "dH_c": 19.5, "LOI": 9.0, "ul94": "HB", "Ea": 180.0, "note": "Pyrophoric in fine dust form, reacts violently with nitrogen/CO2"},
    {"name": "Zirconium Metal Dust", "category": "Combustible Metals & Solids", "rho_s": 6520.0, "c_s": 270.0, "k_s": 22.6, "T_pyro": 2128.0, "L_v_kJkg": 6400.0, "dH_c": 12.0, "LOI": 8.5, "ul94": "HB", "Ea": 140.0, "note": "Extremely pyrophoric dry powder, ignites on static spark"},
    {"name": "Lithium Solid Metal", "category": "Combustible Metals & Solids", "rho_s": 534.0, "c_s": 3580.0, "k_s": 85.0, "T_pyro": 454.0, "L_v_kJkg": 1960.0, "dH_c": 43.0, "LOI": 7.0, "ul94": "HB", "Ea": 110.0, "note": "Battery active metal, reacts with moisture to form hydrogen gas"},
    {"name": "Paraffin Candle Wax", "category": "Combustible Metals & Solids", "rho_s": 900.0, "c_s": 2140.0, "k_s": 0.21, "T_pyro": 520.0, "L_v_kJkg": 600.0, "dH_c": 42.0, "LOI": 16.0, "ul94": "HB", "Ea": 120.0, "note": "Solid alkane wax, vaporizes easily into clean flaming wick fire"},
    {"name": "Stearic Acid Solid", "category": "Combustible Metals & Solids", "rho_s": 940.0, "c_s": 2200.0, "k_s": 0.18, "T_pyro": 630.0, "L_v_kJkg": 700.0, "dH_c": 39.5, "LOI": 16.5, "ul94": "HB", "Ea": 130.0, "note": "Saturated fatty acid, melts and fuels sustained flaming"},
    {"name": "Asphalt / Bitumen Binder", "category": "Combustible Metals & Solids", "rho_s": 1030.0, "c_s": 1900.0, "k_s": 0.17, "T_pyro": 650.0, "L_v_kJkg": 1000.0, "dH_c": 40.0, "LOI": 18.0, "ul94": "HB", "Ea": 150.0, "note": "Paving binder, dense heavy black smoke with oil runoff"},
    {"name": "Coal (Bituminous)", "category": "Combustible Metals & Solids", "rho_s": 1350.0, "c_s": 1260.0, "k_s": 0.26, "T_pyro": 670.0, "L_v_kJkg": 2500.0, "dH_c": 30.0, "LOI": 17.0, "ul94": "HB", "Ea": 140.0, "note": "Solid fossil fuel, volatile evolution followed by glowing char"},
    {"name": "Synthetic Leather (PU / PVC Coated)", "category": "Combustible Metals & Solids", "rho_s": 1150.0, "c_s": 1400.0, "k_s": 0.15, "T_pyro": 570.0, "L_v_kJkg": 1600.0, "dH_c": 28.0, "LOI": 21.0, "ul94": "V-2", "Ea": 130.0, "note": "Composite upholstery fabric, plasticizer volatility drives flame spread"},
    {"name": "Natural Bovine Leather", "category": "Combustible Metals & Solids", "rho_s": 1050.0, "c_s": 1500.0, "k_s": 0.16, "T_pyro": 520.0, "L_v_kJkg": 2000.0, "dH_c": 19.0, "LOI": 24.5, "ul94": "V-2", "Ea": 120.0, "note": "Natural collagen protein skin, chars and self-extinguishes slowly"},
]


def _enrich_material(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches raw material entries with SI-consistent physical parameters and
    derived fire-science metrics for compatibility with FireModel:
      - L_v: Latent heat of pyrolysis/gasification in SI units [J/kg] (converted from L_v_kJkg)
      - emissivity: Hemispherical IR surface emissivity [-] (Incropera & DeWitt, Table A.11)
      - thermal_inertia: k * rho * c [W^2*s/(m^4*K^2)] (Quintiere, 2006)
      - thermal_diffusivity_m2s: alpha_s = k / (rho * c) [m^2/s]
      - heat_of_gasification_kJkg: H_g = c_s*(T_pyro - 298.15)/1000 + L_v_kJkg (Tewarson, 2002)
    """
    mat = dict(raw)
    name = mat["name"]
    cat = mat["category"]

    # Unit normalization: FireModel expects L_v in SI [J/kg], while database stores L_v_kJkg [kJ/kg]
    l_v_kj = mat["L_v_kJkg"]
    mat["L_v"] = l_v_kj * 1000.0  # [J/kg]

    # Infrared surface emissivity for Stefan-Boltzmann surface re-radiation
    mat["emissivity"] = CATEGORY_EMISSIVITY_DEFAULTS.get(cat, 0.88)

    # Apply material-specific overrides (Saffire flight materials, PTFE, non-metallic solids)
    if name in MATERIAL_OVERRIDES:
        mat.update(MATERIAL_OVERRIDES[name])

    # Derived thermophysical invariants for diagnostics & UI display
    rho_s, c_s, k_s = mat["rho_s"], mat["c_s"], mat["k_s"]
    mat["thermal_inertia"] = round(k_s * rho_s * c_s, 2)
    mat["thermal_diffusivity_m2s"] = k_s / max(rho_s * c_s, 1.0)
    dT = max(mat["T_pyro"] - 298.15, 0.0)
    mat["heat_of_gasification_kJkg"] = round((c_s * dT) / 1000.0 + l_v_kj, 2)
    mat.setdefault("saffire_validated", False)

    return mat


MATERIALS_118: List[Dict[str, Any]] = [_enrich_material(m) for m in RAW_MATERIALS_118]
MAT_DICT: Dict[str, Dict[str, Any]] = {m["name"]: m for m in MATERIALS_118}
ALL_CATEGORIES: List[str] = sorted({m["category"] for m in MATERIALS_118})

CORE_NASA_NAMES: List[str] = [
    "SIBAL Standard Fabric",
    "PMMA (Acrylic / Plexiglas)",
    "Nomex 450 (Aramid Fabric)",
]


def get_material_by_name(name: str) -> Dict[str, Any]:
    if name not in MAT_DICT:
        raise KeyError(f"Material '{name}' not found in the database.")
    return MAT_DICT[name]


def get_materials_by_category(category: str) -> List[Dict[str, Any]]:
    return [m for m in MATERIALS_118 if m["category"].lower() == category.lower()]
