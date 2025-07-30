import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile
import os
from weasyprint import HTML, CSS
from jinja2 import Template

# Ajoutez ceci tout en haut de votre fichier (après les imports)
st.cache_data.clear()
st.cache_resource.clear()

# Configuration de la page
st.set_page_config(
    page_title="ProspérIA",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour votre bannière Canva exacte
# CSS personnalisé pour votre bannière moderne et élégante
st.image("nouvelle_banniere.png", use_container_width=True)

# Bannière ProsperIA avec design premium et arbre stylisé
# CSS personnalisé pour votre bannière


# Classes de données améliorées
@dataclass
class ClientProfile:
    nom: str
    prenom: str
    age: int
    situation_familiale: str
    enfants: int
    profession: str
    secteur_activite: str
    revenu_annuel: float
    revenus_passifs: float
    charges_mensuelles: float
    objectifs: List[str]
    objectifs_details: Dict[str, str]
    delai_objectif: int
    tolerance_risque: str
    experience_investissement: str
    profil_investisseur: str
    email: str


@dataclass
class PatrimoineData:
    epargne_courante: float = 0
    livret_a: float = 0
    ldds: float = 0
    cel: float = 0
    pel: float = 0
    assurance_vie_euro: float = 0
    assurance_vie_uc: float = 0
    pea: float = 0
    pea_pme: float = 0
    cto: float = 0
    per_individuel: float = 0
    per_entreprise: float = 0
    crypto: float = 0
    or_physique: float = 0
    scpi: float = 0
    fonds_euros: float = 0
    immobilier_residence: float = 0
    immobilier_locatif: float = 0
    credit_immobilier: float = 0
    credit_conso: float = 0
    autres_dettes: float = 0
    pret_famille: float = 0


@dataclass
class ProfileType:
    nom: str
    age_min: int
    age_max: int
    caracteristiques: List[str]
    allocation_actions: float
    allocation_obligations: float
    allocation_monetaire: float
    allocation_alternatif: float
    recommandations_specifiques: List[str]


# Profils types améliorés
PROFILS_TYPES = {
    "jeune_actif": ProfileType(
        "Jeune Actif Dynamique",
        22, 35,
        ["Constitution patrimoine", "Prise de risque élevée", "Horizon long terme"],
        70, 15, 10, 5,
        ["Maximiser PEA", "Assurance-vie UC", "Éviter sur-liquidité", "SCPI pour diversification"]
    ),
    "cadre_famille": ProfileType(
        "Cadre avec Famille",
        30, 50,
        ["Sécurisation patrimoine", "Préparation avenir enfants", "Équilibre risque/sécurité"],
        50, 25, 20, 5,
        ["Renforcer épargne précaution", "Assurance-vie mixte", "PER déduction fiscale", "Assurance décès"]
    ),
    "pre_retraite": ProfileType(
        "Pré-retraité Prudent",
        50, 65,
        ["Préservation capital", "Préparation retraite", "Réduction progressive risque"],
        30, 40, 25, 5,
        ["Sécurisation progressive", "Fonds euros", "PER optimal", "Stratégie transmission"]
    ),
    "retraite": ProfileType(
        "Retraité Conservateur",
        65, 100,
        ["Préservation capital", "Revenus réguliers", "Transmission patrimoine"],
        15, 35, 45, 5,
        ["Privilégier fonds euros", "Rentes viagères", "Démembrement propriété", "Optimisation transmission"]
    )
}

def generer_points_forts(ratios: Dict, client: ClientProfile) -> List[str]:
    """Génère les points forts du profil patrimonial"""
    points = []

    if ratios['ratio_liquidite'] >= 3:
        points.append(f"Épargne de précaution bien constituée ({ratios['ratio_liquidite']:.1f} mois)")

    if ratios['taux_epargne'] > 0.15:
        points.append(f"Excellente capacité d'épargne ({ratios['taux_epargne']*100:.0f}% des revenus)")

    if ratios['diversification_supports'] > 0.6:
        points.append("Bonne diversification des supports d'épargne")

    if ratios['ratio_endettement'] < 0.3:
        points.append("Niveau d'endettement maîtrisé")

    if client.age < 40 and ratios['exposition_risque'] > 0.4:
        points.append("Allocation dynamique adaptée à votre âge")

    if ratios['utilisation_pea'] > 0.3:
        points.append("Bonne utilisation de l'enveloppe PEA")

    if len(points) == 0:
        points.append("Potentiel d'optimisation important identifié")

    return points

def generer_points_attention(ratios: Dict, client: ClientProfile) -> List[str]:
    """Génère les points d'attention du profil patrimonial"""
    points = []

    if ratios['ratio_liquidite'] < 2:
        points.append("Épargne de précaution insuffisante (risque en cas d'imprévu)")

    if ratios['exposition_risque'] < 0.2 and client.age < 50:
        points.append("Sous-exposition aux marchés financiers (opportunité de croissance manquée)")

    if ratios['diversification_supports'] < 0.4:
        points.append("Concentration excessive sur peu de supports d'épargne")

    if ratios['ratio_endettement'] > 0.5:
        points.append("Niveau d'endettement élevé nécessitant une vigilance")

    if ratios['utilisation_pea'] < 0.2 and client.age < 60:
        points.append("Sous-utilisation de l'enveloppe PEA (avantage fiscal non exploité)")

    if ratios['part_immobilier'] > 0.7:
        points.append("Concentration excessive sur l'immobilier (manque de liquidité)")

    if len(points) == 0:
        points.append("Profil patrimonial globalement équilibré")

    return points


# Fonctions pour sauvegarder les données
def sauvegarder_client():
    """Sauvegarde les données client en session"""
    if all([st.session_state.get('nom'), st.session_state.get('prenom'), st.session_state.get('email')]):
        client = ClientProfile(
            nom=st.session_state.get('nom', ''),
            prenom=st.session_state.get('prenom', ''),
            age=st.session_state.get('age', 35),
            situation_familiale=st.session_state.get('situation', 'Célibataire'),
            enfants=st.session_state.get('enfants', 0),
            profession=st.session_state.get('profession', ''),
            secteur_activite=st.session_state.get('secteur', ''),
            revenu_annuel=st.session_state.get('revenu_annuel', 50000),
            revenus_passifs=st.session_state.get('revenus_passifs', 0),
            charges_mensuelles=st.session_state.get('charges_mensuelles', 2500),
            objectifs=st.session_state.get('objectifs', []),
            objectifs_details=st.session_state.get('objectifs_details', {}),
            delai_objectif=st.session_state.get('delai_objectif', 10),
            tolerance_risque=st.session_state.get('tolerance_risque', 'Modéré'),
            experience_investissement=st.session_state.get('experience_investissement', 'Débutant'),
            profil_investisseur=st.session_state.get('profil_investisseur', 'Équilibré'),
            email=st.session_state.get('email', '')
        )

        st.session_state.client = client
        st.session_state.progress = 25
        return True
    return False


def sauvegarder_patrimoine():
    """Sauvegarde les données patrimoine en session"""
    patrimoine = PatrimoineData(
        epargne_courante=st.session_state.get('epargne_courante', 0),
        livret_a=st.session_state.get('livret_a', 0),
        ldds=st.session_state.get('ldds', 0),
        cel=st.session_state.get('cel', 0),
        pel=st.session_state.get('pel', 0),
        assurance_vie_euro=st.session_state.get('assurance_vie_euro', 0),
        assurance_vie_uc=st.session_state.get('assurance_vie_uc', 0),
        pea=st.session_state.get('pea', 0),
        pea_pme=st.session_state.get('pea_pme', 0),
        cto=st.session_state.get('cto', 0),
        per_individuel=st.session_state.get('per_individuel', 0),
        per_entreprise=st.session_state.get('per_entreprise', 0),
        crypto=st.session_state.get('crypto', 0),
        or_physique=st.session_state.get('or_physique', 0),
        scpi=st.session_state.get('scpi', 0),
        fonds_euros=st.session_state.get('fonds_euros', 0),
        immobilier_residence=st.session_state.get('immobilier_residence', 0),
        immobilier_locatif=st.session_state.get('immobilier_locatif', 0),
        credit_immobilier=st.session_state.get('credit_immobilier', 0),
        credit_conso=st.session_state.get('credit_conso', 0),
        autres_dettes=st.session_state.get('autres_dettes', 0),
        pret_famille=st.session_state.get('pret_famille', 0)
    )

    st.session_state.patrimoine = patrimoine
    st.session_state.progress = 50
    return True


def creer_graphique_repartition_avance(patrimoine: PatrimoineData) -> go.Figure:
    """Crée un graphique de répartition patrimoniale avancé"""

    # Calcul des catégories
    liquidites = patrimoine.epargne_courante + patrimoine.livret_a + patrimoine.ldds + patrimoine.cel
    epargne_reglementee = patrimoine.pel
    assurance_vie = patrimoine.assurance_vie_euro + patrimoine.assurance_vie_uc
    pea_total = patrimoine.pea + patrimoine.pea_pme
    per_total = patrimoine.per_individuel + patrimoine.per_entreprise
    immobilier = patrimoine.immobilier_residence + patrimoine.immobilier_locatif
    alternatifs = patrimoine.crypto + patrimoine.or_physique + patrimoine.scpi

    categories = ['Liquidités', 'Épargne Réglementée', 'Assurance-Vie', 'PEA', 'CTO', 'PER', 'Immobilier',
                  'Alternatifs']
    valeurs = [liquidites, epargne_reglementee, assurance_vie, pea_total, patrimoine.cto, per_total, immobilier,
               alternatifs]

    # Filtrer les valeurs nulles
    categories_filtered = [cat for cat, val in zip(categories, valeurs) if val > 0]
    valeurs_filtered = [val for val in valeurs if val > 0]

    if not valeurs_filtered:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée patrimoniale", x=0.5, y=0.5, showarrow=False)
        return fig

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig = go.Figure(data=[go.Pie(
        labels=categories_filtered,
        values=valeurs_filtered,
        hole=0.4,
        marker_colors=colors[:len(categories_filtered)],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Montant: %{value:,.0f} €<br>Part: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title="🏛️ Répartition Patrimoniale Détaillée",
        showlegend=True,
        height=400,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )

    return fig


def creer_graphique_radar_avance(ratios: Dict, profil_type: ProfileType) -> go.Figure:
    """Crée un graphique radar comparant la situation actuelle au profil cible"""

    categories = ['Liquidité<br>(3-6 mois)', 'Épargne<br>(>15%)', 'Diversification<br>(>60%)',
                  'Exposition Risque<br>(profil)', 'Optimisation<br>Fiscale']

    # Normalisation des valeurs sur 100
    valeurs_actuelles = [
        min(100, ratios['ratio_liquidite'] * 20),  # 5 mois = 100%
        min(100, ratios['taux_epargne'] * 500),  # 20% = 100%
        ratios['diversification_supports'] * 100,
        100 - abs(ratios['exposition_risque'] - profil_type.allocation_actions / 100) * 200,  # Écart à la cible
        (ratios['utilisation_pea'] + ratios['utilisation_av']) * 50  # Utilisation enveloppes
    ]

    # Profil cible (optimal)
    valeurs_cibles = [80, 80, 80, 100, 80]

    fig = go.Figure()

    # Situation actuelle
    fig.add_trace(go.Scatterpolar(
        r=valeurs_actuelles,
        theta=categories,
        fill='toself',
        name='Situation Actuelle',
        line=dict(color='#2a5298', width=2),
        fillcolor='rgba(42, 82, 152, 0.25)'
    ))

    # Profil cible
    fig.add_trace(go.Scatterpolar(
        r=valeurs_cibles,
        theta=categories,
        fill='toself',
        name='Profil Optimal',
        line=dict(color='#28a745', width=2, dash='dash'),
        fillcolor='rgba(40, 167, 69, 0.15)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%')
        ),
        showlegend=True,
        title="🎯 Analyse Comparative - Profil Patrimonial",
        height=400
    )

    return fig


def simuler_monte_carlo_avance(patrimoine_initial: float, versements_annuels: float,
                               rendement_moyen: float, volatilite: float, annees: int,
                               inflation: float, nb_simulations: int = 1000) -> Dict:
    """Simulation Monte Carlo avancée avec versements et inflation"""

    np.random.seed(42)
    resultats = []

    for _ in range(nb_simulations):
        patrimoine = patrimoine_initial
        historique = [patrimoine]

        for annee in range(annees):
            # Rendement aléatoire avec volatilité
            rendement_annuel = np.random.normal(rendement_moyen, volatilite)

            # Application du rendement
            patrimoine *= (1 + rendement_annuel)

            # Ajout des versements (ajustés de l'inflation)
            versement_reel = versements_annuels * ((1 + inflation) ** annee)
            patrimoine += versement_reel

            historique.append(patrimoine)

        resultats.append(historique)

    # Calcul des percentiles
    resultats_array = np.array(resultats)
    percentiles = {
        'p5': np.percentile(resultats_array, 5, axis=0),
        'p25': np.percentile(resultats_array, 25, axis=0),
        'p50': np.percentile(resultats_array, 50, axis=0),
        'p75': np.percentile(resultats_array, 75, axis=0),
        'p95': np.percentile(resultats_array, 95, axis=0)
    }

    return {
        'annees': list(range(annees + 1)),
        'percentiles': percentiles,
        'moyenne': np.mean(resultats_array, axis=0)
    }


def creer_graphique_projections_avance(projections: Dict, horizon: int) -> go.Figure:
    """Crée un graphique avancé des projections"""

    fig = go.Figure()

    couleurs = {"Conservateur": "#28a745", "Équilibré": "#ffc107", "Dynamique": "#dc3545"}

    for nom, proj in projections.items():
        couleur = couleurs[nom]

        # Médiane
        fig.add_trace(go.Scatter(
            x=proj['annees'],
            y=proj['percentiles']['p50'],
            mode='lines',
            name=f"{nom} (Médiane)",
            line=dict(color=couleur, width=3),
            hovertemplate=f'<b>{nom}</b><br>Année: %{{x}}<br>Patrimoine: %{{y:,.0f}} €<extra></extra>'
        ))

        # Zone de confiance 25-75%
        fig.add_trace(go.Scatter(
            x=proj['annees'] + proj['annees'][::-1],
            y=list(proj['percentiles']['p75']) + list(proj['percentiles']['p25'][::-1]),
            fill='tonexty',
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(couleur)) + [0.2])}",
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{nom} (25-75%)",
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"🎯 Projections Patrimoniales sur {horizon} ans - Monte Carlo",
        xaxis_title="Années",
        yaxis_title="Patrimoine (€)",
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig

# Moteur d'analyse patrimoniale intelligent
class AnalyseurPatrimoine:
    def __init__(self, client: ClientProfile, patrimoine: PatrimoineData):
        self.client = client
        self.patrimoine = patrimoine
        self.profil_type = self._determiner_profil_type()

    def _determiner_profil_type(self) -> ProfileType:
        """Détermine le profil type du client"""
        age = self.client.age

        if age <= 35:
            return PROFILS_TYPES["jeune_actif"]
        elif age <= 50:
            return PROFILS_TYPES["cadre_famille"]
        elif age <= 65:
            return PROFILS_TYPES["pre_retraite"]
        else:
            return PROFILS_TYPES["retraite"]

    def calculer_ratios_avances(self) -> Dict:
        """Calcule les ratios financiers avancés"""
        # Calculs patrimoniaux
        liquidites = (self.patrimoine.epargne_courante + self.patrimoine.livret_a +
                      self.patrimoine.ldds + self.patrimoine.cel)

        epargne_reglementee = (self.patrimoine.livret_a + self.patrimoine.ldds +
                               self.patrimoine.cel + self.patrimoine.pel)

        assurance_vie_total = self.patrimoine.assurance_vie_euro + self.patrimoine.assurance_vie_uc

        enveloppes_fiscales = (self.patrimoine.pea + self.patrimoine.pea_pme +
                               assurance_vie_total + self.patrimoine.per_individuel +
                               self.patrimoine.per_entreprise)

        investissements_risques = (self.patrimoine.pea + self.patrimoine.pea_pme +
                                   self.patrimoine.cto + self.patrimoine.assurance_vie_uc +
                                   self.patrimoine.crypto + self.patrimoine.scpi)

        immobilier_total = self.patrimoine.immobilier_residence + self.patrimoine.immobilier_locatif

        dettes_totales = (self.patrimoine.credit_immobilier + self.patrimoine.credit_conso +
                          self.patrimoine.autres_dettes + self.patrimoine.pret_famille)

        patrimoine_financier = (liquidites + epargne_reglementee + enveloppes_fiscales +
                                self.patrimoine.crypto + self.patrimoine.or_physique)

        patrimoine_brut = patrimoine_financier + immobilier_total
        patrimoine_net = patrimoine_brut - dettes_totales

        revenus_totaux = self.client.revenu_annuel + self.client.revenus_passifs
        charges_annuelles = self.client.charges_mensuelles * 12
        capacite_epargne = revenus_totaux - charges_annuelles

        return {
            # Patrimoine
            'patrimoine_brut': patrimoine_brut,
            'patrimoine_net': patrimoine_net,
            'patrimoine_financier': patrimoine_financier,
            'liquidites': liquidites,
            'enveloppes_fiscales': enveloppes_fiscales,
            'investissements_risques': investissements_risques,

            # Ratios clés
            'ratio_patrimoine_revenu': patrimoine_net / revenus_totaux if revenus_totaux > 0 else 0,
            'ratio_liquidite': liquidites / (revenus_totaux / 12) if revenus_totaux > 0 else 0,
            'ratio_endettement': dettes_totales / patrimoine_brut if patrimoine_brut > 0 else 0,
            'taux_epargne': capacite_epargne / revenus_totaux if revenus_totaux > 0 else 0,

            # Diversification
            'diversification_supports': self._calculer_diversification_supports(),
            'exposition_risque': investissements_risques / patrimoine_brut if patrimoine_brut > 0 else 0,
            'part_immobilier': immobilier_total / patrimoine_brut if patrimoine_brut > 0 else 0,

            # Optimisation fiscale
            'utilisation_pea': min(1, (self.patrimoine.pea + self.patrimoine.pea_pme) / 150000),
            'utilisation_av': min(1, assurance_vie_total / 150000),
            'potentiel_per': min(revenus_totaux * 0.1, 32909) if revenus_totaux > 50000 else 0,
        }

    def _calculer_diversification_supports(self) -> float:
        """Calcule le score de diversification des supports"""
        supports = [
            self.patrimoine.epargne_courante,
            self.patrimoine.livret_a + self.patrimoine.ldds,
            self.patrimoine.assurance_vie_euro + self.patrimoine.assurance_vie_uc,
            self.patrimoine.pea + self.patrimoine.pea_pme,
            self.patrimoine.cto,
            self.patrimoine.per_individuel + self.patrimoine.per_entreprise,
            self.patrimoine.scpi,
            self.patrimoine.immobilier_residence + self.patrimoine.immobilier_locatif
        ]

        supports_utilises = sum(1 for s in supports if s > 1000)  # Seuil minimal
        return supports_utilises / len(supports)

    def generer_recommandations_expertes(self) -> List[Dict]:
        """Génère des recommandations d'expert avec logique métier avancée"""
        ratios = self.calculer_ratios_avances()
        recommandations = []

        # 1. Épargne de précaution (priorité absolue)
        if ratios['ratio_liquidite'] < 3:
            montant_manquant = max(0, (self.client.revenu_annuel / 4) - ratios['liquidites'])
            recommandations.append({
                'type': 'URGENT',
                'categorie': 'Sécurisation',
                'titre': 'Constituer votre épargne de précaution',
                'analyse': f"Votre épargne liquide couvre {ratios['ratio_liquidite']:.1f} mois de revenus. La norme prudentielle est de 3 à 6 mois pour faire face aux aléas.",
                'action': f"Épargner {montant_manquant:,.0f} € supplémentaires sur Livret A/LDDS",
                'benefice': "Sécurité financière et sérénité face aux imprévus",
                'priorite': 10,
                'impact_financier': montant_manquant * 0.03,  # Rendement épargne réglementée
                'delai': "0-3 mois"
            })

        # 2. Optimisation enveloppes fiscales
        if ratios['utilisation_pea'] < 0.5 and self.client.age < 60:
            recommandations.append({
                'type': 'STRATEGIQUE',
                'categorie': 'Optimisation Fiscale',
                'titre': 'Maximiser votre Plan d\'Épargne en Actions',
                'analyse': f"Votre PEA n'est utilisé qu'à {ratios['utilisation_pea'] * 100:.0f}% de sa capacité (150k€). Exonération fiscale après 5 ans.",
                'action': f"Programmer des versements mensuels de {min(1000, (150000 - self.patrimoine.pea - self.patrimoine.pea_pme) / 24):,.0f} € sur 2 ans",
                'benefice': "Croissance défiscalisée sur actions européennes",
                'priorite': 8,
                'impact_financier': 50000 * 0.07 * 0.172,  # Économie fiscale estimée
                'delai': "3-24 mois"
            })

        # 3. Rééquilibrage allocation selon profil
        exposition_cible = self.profil_type.allocation_actions / 100
        if abs(ratios['exposition_risque'] - exposition_cible) > 0.15:
            recommandations.append({
                'type': 'REEQUILIBRAGE',
                'categorie': 'Allocation Stratégique',
                'titre': 'Ajuster votre exposition au risque',
                'analyse': f"Votre exposition actuelle ({ratios['exposition_risque'] * 100:.0f}%) s'écarte de votre profil cible ({exposition_cible * 100:.0f}%).",
                'action': "Procéder à un rééquilibrage progressif sur 6 mois" if ratios[
                                                                                     'exposition_risque'] < exposition_cible else "Sécuriser une partie des gains",
                'benefice': "Optimisation rendement/risque selon votre profil",
                'priorite': 7,
                'impact_financier': ratios['patrimoine_brut'] * 0.02,  # Impact estimé
                'delai': "3-6 mois"
            })

        # 4. Optimisation PER pour déduction fiscale
        if ratios['potentiel_per'] > 0 and (self.patrimoine.per_individuel + self.patrimoine.per_entreprise) < ratios[
            'potentiel_per'] * 0.5:
            economie_fiscale = ratios['potentiel_per'] * 0.30  # TMI moyen
            recommandations.append({
                'type': 'FISCALITE',
                'categorie': 'Optimisation Fiscale',
                'titre': 'Optimiser votre déduction fiscale via PER',
                'analyse': f"Avec {self.client.revenu_annuel:,.0f} € de revenus, vous pouvez déduire jusqu'à {ratios['potentiel_per']:,.0f} € par an.",
                'action': f"Ouvrir un PER et y verser {min(ratios['potentiel_per'], 10000):,.0f} € avant fin d'année",
                'benefice': f"Économie d'impôt immédiate de {economie_fiscale:,.0f} € + croissance défiscalisée",
                'priorite': 6,
                'impact_financier': economie_fiscale,
                'delai': "Immédiat"
            })

        # 5. Diversification immobilière (SCPI)
        if ratios['part_immobilier'] < 0.15 and ratios['patrimoine_financier'] > 50000:
            recommandations.append({
                'type': 'DIVERSIFICATION',
                'categorie': 'Immobilier Financier',
                'titre': 'Diversifier vers l\'immobilier financier',
                'analyse': "L'immobilier représente une classe d'actifs décoréllée des marchés financiers avec potentiel de revenus réguliers.",
                'action': f"Investir {min(20000, ratios['patrimoine_financier'] * 0.1):,.0f} € en SCPI de rendement via assurance-vie",
                'benefice': "Revenus locatifs réguliers (4-5% nets) + diversification",
                'priorite': 5,
                'impact_financier': 20000 * 0.045,  # Rendement SCPI moyen
                'delai': "1-6 mois"
            })

        # 6. Stratégie transmission (si patrimoine > 500k€)
        if ratios['patrimoine_net'] > 500000 and self.client.enfants > 0:
            recommandations.append({
                'type': 'TRANSMISSION',
                'categorie': 'Stratégie Patrimoniale',
                'titre': 'Organiser la transmission de votre patrimoine',
                'analyse': f"Avec {ratios['patrimoine_net']:,.0f} € de patrimoine, anticiper la transmission optimise la fiscalité successorale.",
                'action': "Étudier donations régulières et/ou démembrement de propriété",
                'benefice': "Optimisation droits de succession + accompagnement famille",
                'priorite': 4,
                'impact_financier': ratios['patrimoine_net'] * 0.15,  # Économie fiscale potentielle
                'delai': "6-12 mois"
            })

        # Tri par priorité décroissante
        recommandations.sort(key=lambda x: x['priorite'], reverse=True)
        return recommandations[:6]


# Template HTML pour le rapport PDF
TEMPLATE_RAPPORT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analyse Patrimoniale - {{ client.prenom }} {{ client.nom }}</title>
    <style>
        @page {
            margin: 2.5cm;
            size: A4;
            @bottom-center {
                content: "Page " counter(page) " sur " counter(pages);
                font-size: 10px;
                color: #666;
            }
        }

        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }

        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 8px;
        }

        .header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 300;
        }

        .header p {
            margin: 10px 0 0 0;
            font-size: 16px;
            opacity: 0.9;
        }

        .section {
            margin-bottom: 40px;
            page-break-inside: avoid;
        }

        .section-title {
            color: #1e3c72;
            border-bottom: 3px solid #2a5298;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: 600;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: #f8f9fa;
            border-left: 4px solid #2a5298;
            padding: 20px;
            border-radius: 5px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .recommendation {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .recommendation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .recommendation-title {
            font-size: 16px;
            font-weight: 600;
            color: #1e3c72;
        }

        .priority-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            color: white;
        }

        .priority-urgent { background-color: #dc3545; }
        .priority-strategique { background-color: #fd7e14; }
        .priority-reequilibrage { background-color: #6f42c1; }
        .priority-fiscalite { background-color: #20c997; }
        .priority-diversification { background-color: #17a2b8; }
        .priority-transmission { background-color: #6c757d; }

        .recommendation-content {
            font-size: 14px;
            line-height: 1.5;
        }

        .recommendation-action {
            background: #e3f2fd;
            border-left: 3px solid #2196f3;
            padding: 10px 15px;
            margin-top: 10px;
            font-weight: 500;
        }

        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .summary-table th,
        .summary-table td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }

        .summary-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }

        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            font-size: 12px;
            color: #6c757d;
            text-align: center;
        }
    </style>
</head>
<body>
    <!-- Page de couverture -->
    <div class="header">
        <h1>ANALYSE PATRIMONIALE PRIVÉ</h1>
        <p>{{ client.prenom }} {{ client.nom }}</p>
        <p>{{ date_rapport }}</p>
    </div>

    <!-- Résumé exécutif -->
    <div class="section">
        <h2 class="section-title">📊 SYNTHÈSE PATRIMONIALE</h2>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.0f"|format(ratios.patrimoine_net) }} €</div>
                <div class="metric-label">Patrimoine Net</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(ratios.ratio_patrimoine_revenu) }}x</div>
                <div class="metric-label">Ratio Patrimoine/Revenus</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(ratios.ratio_liquidite) }} mois</div>
                <div class="metric-label">Épargne de Précaution</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.0f"|format(ratios.exposition_risque * 100) }}%</div>
                <div class="metric-label">Exposition au Risque</div>
            </div>
        </div>

        <table class="summary-table">
            <tr>
                <th>Profil Client</th>
                <td>{{ profil_type.nom }} ({{ client.age }} ans)</td>
            </tr>
            <tr>
                <th>Situation Familiale</th>
                <td>{{ client.situation_familiale }}{% if client.enfants > 0 %}, {{ client.enfants }} enfant(s){% endif %}</td>
            </tr>
            <tr>
                <th>Revenus Annuels</th>
                <td>{{ "%.0f"|format(client.revenu_annuel) }} € nets</td>
            </tr>
            <tr>
                <th>Profil Investisseur</th>
                <td>{{ client.profil_investisseur }}</td>
            </tr>
            <tr>
                <th>Horizon d'Investissement</th>
                <td>{{ client.delai_objectif }} ans</td>
            </tr>
        </table>
    </div>

    <!-- Objectifs patrimoniaux -->
    <div class="section">
        <h2 class="section-title">🎯 OBJECTIFS PATRIMONIAUX</h2>
        <ul>
        {% for objectif in client.objectifs %}
            <li><strong>{{ objectif }}</strong>{% if client.objectifs_details.get(objectif) %} - {{ client.objectifs_details[objectif] }}{% endif %}</li>
        {% endfor %}
        </ul>
    </div>

    <!-- Diagnostic patrimonial -->
    <div class="section">
        <h2 class="section-title">🔍 DIAGNOSTIC PATRIMONIAL</h2>

        <h3>Points Forts Identifiés :</h3>
        <ul>
        {% for point in points_forts %}
            <li>{{ point }}</li>
        {% endfor %}
        </ul>

        <h3>Points d'Attention :</h3>
        <ul>
        {% for point in points_attention %}
            <li>{{ point }}</li>
        {% endfor %}
        </ul>
    </div>

    <!-- Recommandations prioritaires -->
    <div class="section">
        <h2 class="section-title">📋 PLAN D'ACTION RECOMMANDÉ</h2>

        {% for rec in recommandations %}
        <div class="recommendation">
            <div class="recommendation-header">
                <span class="recommendation-title">{{ loop.index }}. {{ rec.titre }}</span>
                <span class="priority-badge priority-{{ rec.type.lower() }}">{{ rec.type }}</span>
            </div>
            <div class="recommendation-content">
                <p><strong>Analyse :</strong> {{ rec.analyse }}</p>
                <div class="recommendation-action">
                    <strong>Action recommandée :</strong> {{ rec.action }}
                </div>
                <p><strong>Bénéfice attendu :</strong> {{ rec.benefice }}</p>
                <p><strong>Impact financier estimé :</strong> {{ "%.0f"|format(rec.impact_financier) }} € | <strong>Délai :</strong> {{ rec.delai }}</p>
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Allocation recommandée -->
    <div class="section">
        <h2 class="section-title">⚖️ ALLOCATION PATRIMONIALE RECOMMANDÉE</h2>

        <p>Selon votre profil <strong>{{ profil_type.nom }}</strong>, voici la répartition conseillée :</p>

        <table class="summary-table">
            <tr>
                <th>Classe d'Actifs</th>
                <th>Allocation Cible</th>
                <th>Allocation Actuelle</th>
                <th>Écart</th>
            </tr>
            <tr>
                <td>Actions / Fonds dynamiques</td>
                <td>{{ "%.0f"|format(profil_type.allocation_actions) }}%</td>
                <td>{{ "%.0f"|format(ratios.exposition_risque * 100) }}%</td>
                <td>{{ "%.0f"|format(profil_type.allocation_actions - (ratios.exposition_risque * 100)) }}%</td>
            </tr>
            <tr>
                <td>Obligations / Fonds obligataires</td>
                <td>{{ "%.0f"|format(profil_type.allocation_obligations) }}%</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Monétaire / Fonds euros</td>
                <td>{{ "%.0f"|format(profil_type.allocation_monetaire) }}%</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Alternatifs (SCPI, Or...)</td>
                <td>{{ "%.0f"|format(profil_type.allocation_alternatif) }}%</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </table>
    </div>

    <!-- Calendrier de mise en œuvre -->
    <div class="section">
        <h2 class="section-title">📅 CALENDRIER DE MISE EN ŒUVRE</h2>

        <h3>Phase 1 - Actions Immédiates (0-3 mois) :</h3>
        <ul>
        {% for rec in recommandations %}
            {% if rec.delai in ["Immédiat", "0-3 mois"] %}
            <li>{{ rec.titre }}</li>
            {% endif %}
        {% endfor %}
        </ul>

        <h3>Phase 2 - Structuration (3-12 mois) :</h3>
        <ul>
        {% for rec in recommandations %}
            {% if "mois" in rec.delai and rec.delai not in ["Immédiat", "0-3 mois"] %}
            <li>{{ rec.titre }}</li>
            {% endif %}
        {% endfor %}
        </ul>

        <h3>Phase 3 - Optimisation Continue (12+ mois) :</h3>
        <ul>
            <li>Rééquilibrage semestriel du portefeuille</li>
            <li>Optimisation fiscale annuelle</li>
            <li>Révision stratégie selon évolutions personnelles</li>
        </ul>
    </div>

    <div class="footer">
        <p><strong>Rapport généré par PatrimoineAnalyzer Pro</strong></p>
        <p>Cette analyse est fournie à titre informatif et ne constitue pas un conseil en investissement personnalisé.<br>
        Il est recommandé de consulter un conseiller en gestion de patrimoine avant toute décision d'investissement.</p>
        <p>© {{ date_rapport.split()[2] }} - PatrimoineAnalyzer Pro - Analyse patrimoniale intelligente</p>
    </div>
</body>
</html>
"""


def generer_rapport_html(client: ClientProfile, patrimoine: PatrimoineData,
                         analyseur: AnalyseurPatrimoine) -> str:
    """Génère un rapport HTML professionnel complet"""

    ratios = analyseur.calculer_ratios_avances()
    recommandations = analyseur.generer_recommandations_expertes()
    points_forts = generer_points_forts(ratios, client)
    points_attention = generer_points_attention(ratios, client, patrimoine)

    # Calculs détaillés pour le rapport
    liquidites = patrimoine.epargne_courante + patrimoine.livret_a + patrimoine.ldds + patrimoine.cel
    epargne_reglementee = patrimoine.pel + patrimoine.fonds_euros
    assurance_vie_total = patrimoine.assurance_vie_euro + patrimoine.assurance_vie_uc
    pea_total = patrimoine.pea + patrimoine.pea_pme
    per_total = patrimoine.per_individuel + patrimoine.per_entreprise
    immobilier_total = patrimoine.immobilier_residence + patrimoine.immobilier_locatif
    alternatifs = patrimoine.crypto + patrimoine.or_physique + patrimoine.scpi

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Analyse Patrimoniale - {client.prenom} {client.nom}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f8f9fa;
                color: #333;
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                border-radius: 10px;
                overflow: hidden;
            }}
            .header {{ 
                background: linear-gradient(135deg, #2B3544 0%, #1e3c72 100%); 
                color: white; 
                padding: 40px; 
                text-align: center; 
            }}
            .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
            .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
            .content {{ padding: 30px; }}
            .section {{ 
                margin: 40px 0; 
                padding-bottom: 30px; 
                border-bottom: 2px solid #e9ecef;
            }}
            .section:last-child {{ border-bottom: none; }}
            .section h2 {{ 
                color: #2B3544; 
                font-size: 1.8em; 
                margin-bottom: 20px; 
                padding-bottom: 10px; 
                border-bottom: 3px solid #007bff;
            }}
            .metrics-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }}
            .metric {{ 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                text-align: center;
                border-left: 4px solid #007bff;
            }}
            .metric-value {{ 
                font-size: 1.8em; 
                font-weight: bold; 
                color: #2B3544; 
                margin-bottom: 5px; 
            }}
            .metric-label {{ 
                font-size: 0.9em; 
                color: #666; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
            }}
            .recommendation {{ 
                background: #fff; 
                border: 1px solid #e9ecef; 
                border-radius: 8px; 
                padding: 20px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .recommendation h4 {{ 
                color: #2B3544; 
                margin: 0 0 10px 0; 
                font-size: 1.2em; 
            }}
            .recommendation .type {{ 
                display: inline-block; 
                padding: 4px 12px; 
                border-radius: 12px; 
                font-size: 0.8em; 
                font-weight: bold; 
                color: white; 
                margin-bottom: 10px;
            }}
            .urgent {{ background-color: #dc3545; }}
            .strategique {{ background-color: #fd7e14; }}
            .reequilibrage {{ background-color: #6f42c1; }}
            .fiscalite {{ background-color: #20c997; }}
            .diversification {{ background-color: #17a2b8; }}
            .transmission {{ background-color: #6c757d; }}
            .points-list {{ 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 15px 0; 
            }}
            .points-list ul {{ margin: 0; padding-left: 20px; }}
            .points-list li {{ margin-bottom: 8px; line-height: 1.5; }}
            .patrimoine-table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }}
            .patrimoine-table th, .patrimoine-table td {{ 
                border: 1px solid #dee2e6; 
                padding: 12px; 
                text-align: left; 
            }}
            .patrimoine-table th {{ 
                background-color: #2B3544; 
                color: white; 
                font-weight: 600; 
            }}
            .patrimoine-table tr:nth-child(even) {{ 
                background-color: #f8f9fa; 
            }}
            .alert {{ 
                padding: 15px; 
                border-radius: 8px; 
                margin: 20px 0; 
            }}
            .alert-info {{ 
                background-color: #d1ecf1; 
                border-left: 4px solid #17a2b8; 
                color: #0c5460; 
            }}
            .footer {{ 
                background: #2B3544; 
                color: white; 
                padding: 20px; 
                text-align: center; 
                font-size: 0.9em; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ANALYSE PATRIMONIALE PRIVÉ</h1>
                <p>{client.prenom} {client.nom}</p>
                <p>{datetime.now().strftime('%d %B %Y')}</p>
            </div>

            <div class="content">
                <!-- SYNTHÈSE EXÉCUTIVE -->
                <div class="section">
                    <h2>📊 SYNTHÈSE EXÉCUTIVE</h2>

                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">{ratios['patrimoine_net']:,.0f} €</div>
                            <div class="metric-label">Patrimoine Net</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{analyseur.profil_type.nom}</div>
                            <div class="metric-label">Profil Investisseur</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{ratios['ratio_liquidite']:.1f} mois</div>
                            <div class="metric-label">Épargne de Précaution</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{ratios['exposition_risque'] * 100:.0f}%</div>
                            <div class="metric-label">Exposition au Risque</div>
                        </div>
                    </div>

                    <table class="patrimoine-table">
                        <tr><th>Situation Familiale</th><td>{client.situation_familiale}{''.join([f', {client.enfants} enfant(s)' if client.enfants > 0 else ''])}</td></tr>
                        <tr><th>Âge</th><td>{client.age} ans</td></tr>
                        <tr><th>Profession</th><td>{client.profession} - {client.secteur_activite}</td></tr>
                        <tr><th>Revenus Annuels</th><td>{client.revenu_annuel:,.0f} € nets</td></tr>
                        <tr><th>Capacité d'Épargne</th><td>{ratios['taux_epargne'] * 100:.0f}% des revenus</td></tr>
                        <tr><th>Horizon d'Investissement</th><td>{client.delai_objectif} ans</td></tr>
                    </table>
                </div>

                <!-- OBJECTIFS PATRIMONIAUX -->
                <div class="section">
                    <h2>🎯 OBJECTIFS PATRIMONIAUX</h2>
                    <div class="points-list">
                        <ul>
                        {''.join([f'<li><strong>{objectif}</strong></li>' for objectif in client.objectifs])}
                        </ul>
                    </div>
                </div>

                <!-- RÉPARTITION PATRIMONIALE -->
                <div class="section">
                    <h2>💼 RÉPARTITION PATRIMONIALE DÉTAILLÉE</h2>

                    <table class="patrimoine-table">
                        <tr><th>Catégorie</th><th>Montant</th><th>Pourcentage</th></tr>
                        <tr><td>💧 Liquidités (comptes, livrets)</td><td>{liquidites:,.0f} €</td><td>{liquidites / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>🏦 Épargne Réglementée</td><td>{epargne_reglementee:,.0f} €</td><td>{epargne_reglementee / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>🛡️ Assurance-Vie</td><td>{assurance_vie_total:,.0f} €</td><td>{assurance_vie_total / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>📈 PEA / PEA-PME</td><td>{pea_total:,.0f} €</td><td>{pea_total / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>📊 Compte-Titres</td><td>{patrimoine.cto:,.0f} €</td><td>{patrimoine.cto / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>🏛️ PER (retraite)</td><td>{per_total:,.0f} €</td><td>{per_total / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>🏠 Immobilier</td><td>{immobilier_total:,.0f} €</td><td>{immobilier_total / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr><td>💎 Alternatifs (SCPI, crypto, or)</td><td>{alternatifs:,.0f} €</td><td>{alternatifs / ratios['patrimoine_brut'] * 100:.1f}%</td></tr>
                        <tr style="background-color: #2B3544; color: white; font-weight: bold;">
                            <td>TOTAL PATRIMOINE BRUT</td><td>{ratios['patrimoine_brut']:,.0f} €</td><td>100%</td>
                        </tr>
                    </table>
                </div>

                <!-- DIAGNOSTIC PATRIMONIAL -->
                <div class="section">
                    <h2>🔍 DIAGNOSTIC PATRIMONIAL</h2>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                        <div>
                            <h3 style="color: #28a745;">✅ Points Forts Identifiés</h3>
                            <div class="points-list">
                                <ul>
                                {''.join([f'<li>{point}</li>' for point in points_forts])}
                                </ul>
                            </div>
                        </div>

                        <div>
                            <h3 style="color: #dc3545;">⚠️ Points d'Attention</h3>
                            <div class="points-list">
                                <ul>
                                {''.join([f'<li>{point}</li>' for point in points_attention])}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- RECOMMANDATIONS PRIORITAIRES -->
                <div class="section">
                    <h2>🎯 PLAN D'ACTION RECOMMANDÉ</h2>

                    <div class="alert alert-info">
                        <strong>📈 Impact financier total estimé :</strong> {sum([r['impact_financier'] for r in recommandations]):,.0f} € par an
                    </div>

                    {''.join([f'''
                    <div class="recommendation">
                        <h4>{i + 1}. {rec['titre']}</h4>
                        <span class="type {rec['type'].lower()}">{rec['type']}</span>

                        <p><strong>🔍 Analyse :</strong> {rec['analyse']}</p>

                        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%); 
                                   border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0;">
                            <strong>🎯 Action recommandée :</strong><br>
                            {rec['action']}
                        </div>

                        <p><strong>✨ Bénéfice attendu :</strong> <span style="color: #28a745; font-weight: 500;">{rec['benefice']}</span></p>

                        <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666; margin-top: 15px;">
                            <span><strong>💰 Impact :</strong> {rec['impact_financier']:,.0f} € / an</span>
                            <span><strong>⏱️ Délai :</strong> {rec['delai']}</span>
                            <span><strong>🎯 Priorité :</strong> {rec['priorite']}/10</span>
                        </div>
                    </div>
                    ''' for i, rec in enumerate(recommandations)])}
                </div>

                <!-- ALLOCATION RECOMMANDÉE -->
                <div class="section">
                    <h2>⚖️ ALLOCATION PATRIMONIALE RECOMMANDÉE</h2>

                    <p>Selon votre profil <strong>{analyseur.profil_type.nom}</strong>, voici la répartition conseillée :</p>

                    <table class="patrimoine-table">
                        <tr><th>Classe d'Actifs</th><th>Allocation Cible</th><th>Allocation Actuelle</th><th>Écart</th></tr>
                        <tr>
                            <td>📈 Actions / Fonds dynamiques</td>
                            <td>{analyseur.profil_type.allocation_actions:.0f}%</td>
                            <td>{ratios['exposition_risque'] * 100:.0f}%</td>
                            <td style="color: {'#28a745' if ratios['exposition_risque'] * 100 >= analyseur.profil_type.allocation_actions else '#dc3545'};">
                                {ratios['exposition_risque'] * 100 - analyseur.profil_type.allocation_actions:+.0f}%
                            </td>
                        </tr>
                        <tr>
                            <td>🏦 Obligations / Fonds obligataires</td>
                            <td>{analyseur.profil_type.allocation_obligations:.0f}%</td>
                            <td>-</td>
                            <td>À définir</td>
                        </tr>
                        <tr>
                            <td>💰 Monétaire / Fonds euros</td>
                            <td>{analyseur.profil_type.allocation_monetaire:.0f}%</td>
                            <td>-</td>
                            <td>À définir</td>
                        </tr>
                        <tr>
                            <td>💎 Alternatifs (SCPI, Or...)</td>
                            <td>{analyseur.profil_type.allocation_alternatif:.0f}%</td>
                            <td>{alternatifs / ratios['patrimoine_brut'] * 100:.0f}%</td>
                            <td style="color: {'#28a745' if alternatifs / ratios['patrimoine_brut'] * 100 >= analyseur.profil_type.allocation_alternatif else '#dc3545'};">
                                {alternatifs / ratios['patrimoine_brut'] * 100 - analyseur.profil_type.allocation_alternatif:+.0f}%
                            </td>
                        </tr>
                    </table>
                </div>

                <!-- CALENDRIER DE MISE EN ŒUVRE -->
                <div class="section">
                    <!-- CALENDRIER DE MISE EN ŒUVRE -->
<div class="section">
    <h2>📅 CALENDRIER DE MISE EN ŒUVRE</h2>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
        <div>
    <h3 style="color: #dc3545;">🚨 Phase 1 - Actions Prioritaires</h3>
    <div class="points-list">
        <ul>
        {''.join([f'<li>{rec["titre"]}</li>' for rec in recommandations[:max(1, len(recommandations)//2)]])}
        </ul>
    </div>
</div>

<div>
    <h3 style="color: #ffc107;">📈 Phase 2 - Actions Complémentaires</h3>
    <div class="points-list">
        <ul>
        {''.join([f'<li>{rec["titre"]}</li>' for rec in recommandations[len(recommandations)//2:]])}
        {f'<li>Révision et ajustement des stratégies mises en place</li>' if len(recommandations) < 2 else ''}
        </ul>
    </div>
</div>
        
        <div>
            <h3 style="color: #28a745;">🎯 Phase 3 - Optimisation Continue</h3>
            <div class="points-list">
                <ul>
                    <li>Rééquilibrage semestriel du portefeuille</li>
                    <li>Optimisation fiscale annuelle</li>
                    <li>Révision stratégie selon évolutions personnelles</li>
                    <li>Suivi des performances et ajustements</li>
                </ul>
            </div>
        </div>
    </div>
</div>
            </div>

            <div class="footer">
                <p><strong>Rapport généré par ProspérIA - Conseil en Gestion de Patrimoine</strong></p>
                <p>Cette analyse est fournie à titre informatif et ne constitue pas un conseil en investissement personnalisé.<br>
                Il est recommandé de consulter un conseiller en gestion de patrimoine avant toute décision d'investissement.</p>
                <p>© {datetime.now().year} - ProspérIA - Analyse patrimoniale intelligente</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


def generer_points_attention(ratios: Dict, client: ClientProfile, patrimoine: PatrimoineData) -> List[str]:
    """Génère les points d'attention du profil patrimonial"""
    points = []

    if ratios['ratio_liquidite'] < 2:
        points.append("Épargne de précaution insuffisante (risque en cas d'imprévu)")

    if ratios['exposition_risque'] < 0.2 and client.age < 50:
        points.append("Sous-exposition aux marchés financiers (opportunité de croissance manquée)")

    if ratios['diversification_supports'] < 0.4:
        points.append("Concentration excessive sur peu de supports d'épargne")

    if ratios['ratio_endettement'] > 0.5:
        points.append("Niveau d'endettement élevé nécessitant une vigilance")

    if ratios['utilisation_pea'] < 0.2 and client.age < 60:
        points.append("Sous-utilisation de l'enveloppe PEA (avantage fiscal non exploité)")

    if ratios['part_immobilier'] > 0.7:
        points.append("Concentration excessive sur l'immobilier (manque de liquidité)")

    # LIGNE CORRIGÉE : patrimoine au lieu de client.patrimoine
    if ratios['potentiel_per'] > 5000 and (patrimoine.per_individuel + patrimoine.per_entreprise) == 0:
        points.append("Optimisation fiscale PER non exploitée malgré revenus éligibles")

    if len(points) == 0:
        points.append("Profil patrimonial globalement équilibré")

    return points
def generer_points_attention(ratios: Dict, client: ClientProfile, patrimoine: PatrimoineData) -> List[str]:
    """Génère les points d'attention du profil patrimonial"""
    points = []

    if ratios['ratio_liquidite'] < 2:
        points.append("Épargne de précaution insuffisante (risque en cas d'imprévu)")

    if ratios['exposition_risque'] < 0.2 and client.age < 50:
        points.append("Sous-exposition aux marchés financiers (opportunité de croissance manquée)")

    if ratios['diversification_supports'] < 0.4:
        points.append("Concentration excessive sur peu de supports d'épargne")

    if ratios['ratio_endettement'] > 0.5:
        points.append("Niveau d'endettement élevé nécessitant une vigilance")

    if ratios['utilisation_pea'] < 0.2 and client.age < 60:
        points.append("Sous-utilisation de l'enveloppe PEA (avantage fiscal non exploité)")

    if ratios['part_immobilier'] > 0.7:
        points.append("Concentration excessive sur l'immobilier (manque de liquidité)")

    if ratios['potentiel_per'] > 5000 and (patrimoine.per_individuel + patrimoine.per_entreprise) == 0:
        points.append("Optimisation fiscale PER non exploitée malgré revenus éligibles")

    if len(points) == 0:
        points.append("Profil patrimonial globalement équilibré")

    return points


# Fonction d'envoi par email
def envoyer_rapport_email(email_destinataire: str, nom_client: str, html_content: str) -> bool:
    """Envoie le rapport HTML par email"""

    try:
        # Configuration email - MODIFIEZ CES VALEURS
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        email_expediteur = st.secrets.get("EMAIL_SENDER", "votre_email@gmail.com")
        mot_de_passe = st.secrets.get("EMAIL_PASSWORD", "votre_mot_de_passe_app")

        # Si pas de secrets configurés, utiliser des valeurs par défaut (pour test)
        if email_expediteur == "votre_email@gmail.com":
            st.error("⚠️ Veuillez configurer vos paramètres email dans les secrets Streamlit")
            return False

        # Création du message
        msg = MIMEMultipart()
        msg['From'] = email_expediteur
        msg['To'] = email_destinataire
        msg['Subject'] = f"Votre Analyse Patrimoniale - {nom_client}"

        # Corps du message
        corps_message = f"""
        Madame, Monsieur {nom_client},

        Conformément à votre demande, nous vous adressons ci-joint votre analyse patrimoniale complète.

        Cette étude approfondie présente :
        - L'évaluation détaillée de votre patrimoine actuel
        - L'analyse de votre profil de risque et de vos objectifs
        - Nos recommandations stratégiques personnalisées
        - Le plan de mise en œuvre optimisé

        Nous restons à votre entière disposition pour échanger sur ces préconisations ainsi que pour répondre à vos questions.

        Nous vous prions d'agréer, Madame, Monsieur, l'expression de nos salutations distinguées.

        ProspérIA - Conseil en Gestion de Patrimoine
        Département Analyse & Stratégie Patrimoniale
        """

        msg.attach(MIMEText(corps_message, 'plain', 'utf-8'))

        # Pièce jointe HTML
        piece_jointe = MIMEText(html_content, 'html', 'utf-8')
        piece_jointe.add_header(
            'Content-Disposition',
            f'attachment; filename="Analyse_Patrimoniale_{nom_client.replace(" ", "_")}.html"'
        )
        msg.attach(piece_jointe)

        # Envoi
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_expediteur, mot_de_passe)
        server.send_message(msg)
        server.quit()

        return True

    except Exception as e:
        st.error(f"Erreur lors de l'envoi : {str(e)}")
        return False


def envoyer_rapport_par_email(client: ClientProfile, patrimoine: PatrimoineData,
                              analyseur: AnalyseurPatrimoine, email: str, message: str = ""):
    # Configuration email
    smtp_server = "smtp.zoho.eu"
    smtp_port = 587
    email_sender = st.secrets["EMAIL_SENDER"]
    email_password = st.secrets["EMAIL_PASSWORD"]

    try:
        # 1. Génération du HTML complet (votre fonction existante)
        html_content = generer_rapport_html(client, patrimoine, analyseur)

        # 2. Création du message avec HTML
        msg = MIMEMultipart('alternative')
        msg['From'] = email_sender
        msg['To'] = email
        msg['Subject'] = f"Votre Analyse Patrimoniale Professionnelle - {client.prenom} {client.nom}"

        # 3. Message texte de présentation
        texte_presentation = f"""
Bonjour {client.prenom} {client.nom},

Vous trouverez ci-dessous votre analyse patrimoniale personnalisée réalisée par PatrimoineAnalyzer Pro.

Ce rapport complet comprend :
- Une photographie détaillée de votre situation patrimoniale actuelle
- Des recommandations d'expert personnalisées selon votre profil
- Un plan d'action structuré
- Des projections et scénarios d'évolution

{message}

Cordialement,
L'équipe PatrimoineAnalyzer Pro
"""

        # 4. Attachement du texte ET du HTML
        msg.attach(MIMEText(texte_presentation, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))  # ✅ HTML complet

        # 5. Envoi
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_sender, email_password)
        server.send_message(msg)
        server.quit()

        st.success(f"✅ Rapport complet envoyé à {email}")
        return True

    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        return False

# Fonction utilitaire pour activer/désactiver le mode debug
def get_debug_mode():
    # Active le mode debug si nécessaire (mettre True ou False)
    return False


# Interface utilisateur améliorée
def main():
    # Sidebar pour navigation
    with st.sidebar:
        st.markdown("### 🔍 Navigation")
        page = st.selectbox(
            "Choisir une section",
            ["👤 Profil Client", "💼 Patrimoine", "📊 Analyse Experte", "🎯 Recommandations", "📈 Projections",
             "📄 Rapport PDF"],
            key="sidebar_navigation"  # ← Ajoutez cette ligne
        )
        # Barre de progression
        # Barre de progression
        if 'progress' not in st.session_state:
            st.session_state.progress = 0

        st.progress(st.session_state.progress / 100)
        st.caption(f"Progression: {st.session_state.progress}%")

        if st.session_state.progress == 100:
            st.success("✅ Profil complet !")

    # Navigation entre les pages
    if page == "👤 Profil Client":
        page_profil_client()
    elif page == "💼 Patrimoine":
        page_patrimoine_detaille()
    elif page == "📊 Analyse Experte":
        page_analyse_experte()
    elif page == "🎯 Recommandations":
        page_recommandations_ia()
    elif page == "📈 Projections":
        page_projections_avancees()
    elif page == "📄 Rapport PDF":
        page_rapport_professionnel()

def page_profil_client():
    st.header("👤 Profil Client Détaillé")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Informations Personnelles")

        nom = st.text_input("Nom de famille *", key="nom")
        prenom = st.text_input("Prénom *", key="prenom")
        age = st.number_input("Âge", min_value=18, max_value=100, value=35, key="age")

        situation = st.selectbox(
            "Situation familiale",
            ["Célibataire", "En couple", "Marié(e)", "Pacsé(e)", "Divorcé(e)", "Veuf(ve)"],
            key="situation"
        )

        enfants = st.number_input("Nombre d'enfants à charge", min_value=0, max_value=10, value=0, key="enfants")

        email = st.text_input("Email (pour envoi du rapport) *", key="email")

    with col2:
        st.subheader("💼 Situation Professionnelle")

        profession = st.text_input("Profession", key="profession")

        secteur = st.selectbox(
            "Secteur d'activité",
            ["Fonction publique", "Secteur privé", "Libéral", "Entrepreneur", "Retraité", "Étudiant", "Autre"],
            key="secteur"
        )

        revenu_annuel = st.number_input(
            "Revenus nets annuels (€)",
            min_value=0,
            value=50000,
            step=1000,
            key="revenu_annuel"
        )

        revenus_passifs = st.number_input(
            "Revenus passifs annuels (€)",
            min_value=0,
            value=0,
            step=500,
            key="revenus_passifs"
        )

        charges_mensuelles = st.number_input(
            "Charges mensuelles moyennes (€)",
            min_value=0,
            value=2500,
            step=100,
            key="charges_mensuelles"
        )

        # Ajout des objectifs et profil investisseur
        st.subheader("🎯 Objectifs Patrimoniaux")

        objectifs_disponibles = [
            "Constituer une épargne de précaution",
            "Investir en bourse",
            "Préparer la retraite",
            "Financer un projet immobilier",
            "Optimiser la fiscalité",
            "Transmettre un patrimoine"
        ]

        objectifs = st.multiselect("Vos objectifs prioritaires", objectifs_disponibles, key="objectifs")

        tolerance_risque = st.selectbox(
            "Tolérance au risque",
            ["Très prudent", "Prudent", "Modéré", "Dynamique", "Très dynamique"],
            key="tolerance_risque"
        )

        experience_investissement = st.selectbox(
            "Expérience en investissement",
            ["Débutant", "Intermédiaire", "Confirmé", "Expert"],
            key="experience_investissement"
        )

        profil_investisseur = st.selectbox(
            "Profil investisseur",
            ["Conservateur", "Équilibré", "Dynamique"],
            key="profil_investisseur"
        )

        delai_objectif = st.slider("Horizon d'investissement (années)", 1, 30, 10, key="delai_objectif")

        # Bouton de sauvegarde
        if st.button("💾 Sauvegarder le Profil", type="primary"):
            if sauvegarder_client():
                st.success("✅ Profil sauvegardé avec succès!")
                st.balloons()
            else:  # ← CORRIGÉ : même niveau que le if sauvegarder_client()
                st.error("❌ Veuillez remplir tous les champs obligatoires")

        # 🔽 LES FONCTIONS DOIVENT ÊTRE ICI - NIVEAU RACINE (pas d'indentation) 🔽

        def generer_et_telecharger_rapport(client: ClientProfile, patrimoine: PatrimoineData,
                                           analyseur: AnalyseurPatrimoine):
            """Génère et propose le téléchargement du rapport PDF"""

            with st.spinner("📊 Génération du rapport PDF en cours..."):
                try:
                    # Génération du PDF
                    html_content = generer_rapport_html(client, patrimoine, analyseur)

                    # Nom du fichier
                    nom_fichier = f"Analyse_Patrimoniale_{client.prenom}_{client.nom}_{datetime.now().strftime('%Y%m%d')}.pdf"

                    st.success("✅ Rapport généré avec succès!")

                    # Bouton de téléchargement
                    st.download_button(
                        label="📥 Télécharger le Rapport HTML",
                        data=html_content,
                        file_name=nom_fichier,
                        mime="text/html",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération du rapport : {str(e)}")
                    # Statistiques du rapport
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("📄 Pages", "~6-8")
                    with col2:
                        st.metric("📊 Graphiques", "4-6")
                    with col3:
                        st.metric("💾 Taille", f"{len(pdf_content) / 1024:.0f} KB")

                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération du rapport : {str(e)}")
                    st.info("💡 Vérifiez que toutes les données sont correctement saisies.")

        def envoyer_rapport_par_email(client: ClientProfile, patrimoine: PatrimoineData,
                                      analyseur: AnalyseurPatrimoine, email: str, message: str = ""):
            """Envoie le rapport par email"""

            with st.spinner("📧 Envoi du rapport par email..."):
                try:
                    # Génération du HTML
                    html_content = generer_rapport_html(client, patrimoine, analyseur)

                    # Envoi par email
                    success = envoyer_rapport_email(email, f"{client.prenom} {client.nom}", html_content)

                    if success:
                        st.success(f"✅ Rapport envoyé avec succès à {email}")
                        st.balloons()
                    else:
                        st.error("❌ Échec de l'envoi. Vérifiez la configuration email.")

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'envoi : {str(e)}")

        def afficher_apercu_rapport(client: ClientProfile, patrimoine: PatrimoineData, analyseur: AnalyseurPatrimoine):
            """Affiche un aperçu du rapport"""

            st.subheader("👀 Aperçu du Rapport")

            ratios = analyseur.calculer_ratios_avances()
            recommandations = analyseur.generer_recommandations_expertes()

            def simuler_monte_carlo_avance(patrimoine_initial: float, versements_annuels: float,
                                           rendement_moyen: float, volatilite: float, annees: int,
                                           inflation: float, nb_simulations: int = 1000) -> Dict:
                """Simulation Monte Carlo avancée avec versements et inflation"""

                np.random.seed(42)
                resultats = []

                for _ in range(nb_simulations):
                    patrimoine = patrimoine_initial
                    historique = [patrimoine]

                    for annee in range(annees):
                        # Rendement aléatoire avec volatilité
                        rendement_annuel = np.random.normal(rendement_moyen, volatilite)

                        # Application du rendement
                        patrimoine *= (1 + rendement_annuel)

                        # Ajout des versements (ajustés de l'inflation)
                        versement_reel = versements_annuels * ((1 + inflation) ** annee)
                        patrimoine += versement_reel

                        historique.append(patrimoine)

                    resultats.append(historique)

                # Calcul des percentiles
                resultats_array = np.array(resultats)
                percentiles = {
                    'p5': np.percentile(resultats_array, 5, axis=0),
                    'p25': np.percentile(resultats_array, 25, axis=0),
                    'p50': np.percentile(resultats_array, 50, axis=0),
                    'p75': np.percentile(resultats_array, 75, axis=0),
                    'p95': np.percentile(resultats_array, 95, axis=0)
                }

                return {
                    'annees': list(range(annees + 1)),
                    'percentiles': percentiles,
                    'moyenne': np.mean(resultats_array, axis=0)
                }

            def creer_graphique_projections_avance(projections: Dict, horizon: int) -> go.Figure:
                """Crée un graphique avancé des projections"""

                fig = go.Figure()

                couleurs = {"Conservateur": "#28a745", "Équilibré": "#ffc107", "Dynamique": "#dc3545"}

                for nom, proj in projections.items():
                    couleur = couleurs[nom]

                    # Médiane
                    fig.add_trace(go.Scatter(
                        x=proj['annees'],
                        y=proj['percentiles']['p50'],
                        mode='lines',
                        name=f"{nom} (Médiane)",
                        line=dict(color=couleur, width=3),
                        hovertemplate=f'<b>{nom}</b><br>Année: %{{x}}<br>Patrimoine: %{{y:,.0f}} €<extra></extra>'
                    ))

                    # Zone de confiance 25-75%
                    fig.add_trace(go.Scatter(
                        x=proj['annees'] + proj['annees'][::-1],
                        y=list(proj['percentiles']['p75']) + list(proj['percentiles']['p25'][::-1]),
                        fill='tonexty',
                        fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(couleur)) + [0.2])}",
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f"{nom} (25-75%)",
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                fig.update_layout(
                    title=f"🎯 Projections Patrimoniales sur {horizon} ans - Monte Carlo",
                    xaxis_title="Années",
                    yaxis_title="Patrimoine (€)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                )

                return fig

            def page_rapport_professionnel():
                if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
                    st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
                    return

                client = st.session_state.client
                patrimoine = st.session_state.patrimoine
                analyseur = AnalyseurPatrimoine(client, patrimoine)

                st.header("📄 Génération de Rapport Professionnel")

                # Options du rapport
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("⚙️ Configuration du Rapport")

                    inclure_graphiques = st.checkbox("📊 Inclure les graphiques", value=True)
                    inclure_projections = st.checkbox("📈 Inclure les projections", value=True)
                    inclure_benchmarks = st.checkbox("📊 Inclure les benchmarks", value=True)

                    format_email = st.selectbox(
                        "📧 Format d'envoi",
                        ["PDF uniquement", "PDF + Résumé Email", "Résumé Email uniquement"],
                        key="format_email"
                    )

                with col2:
                    st.subheader("📋 Contenu Inclus")

                    elements_inclus = [
                        "✅ Synthèse patrimoniale",
                        "✅ Profil investisseur identifié",
                        "✅ Diagnostic détaillé",
                        "✅ Recommandations prioritaires",
                        "✅ Plan d'action structuré",
                        "✅ Calendrier de mise en œuvre"
                    ]

                    if inclure_projections:
                        elements_inclus.append("✅ Projections Monte Carlo")
                    if inclure_benchmarks:
                        elements_inclus.append("✅ Comparaison benchmarks")

                    for element in elements_inclus:
                        st.markdown(element)

                # Aperçu des métriques clés
                st.subheader("📊 Aperçu du Rapport")

                ratios = analyseur.calculer_ratios_avances()
                recommandations = analyseur.generer_recommandations_expertes()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🏛️ Patrimoine Net", f"{ratios['patrimoine_net']:,.0f} €")

                with col2:
                    st.metric("🎯 Recommandations", len(recommandations))

                with col3:
                    impact_total = sum([r['impact_financier'] for r in recommandations])
                    st.metric("💰 Impact Annuel", f"{impact_total:,.0f} €")

                with col4:
                    st.metric("👤 Profil", analyseur.profil_type.nom)

                # Génération et envoi
                st.markdown("---")

                col1, col2 = st.columns([2, 1])

                with col1:
                    email_destinataire = st.text_input(
                        "📧 Email de réception du rapport",
                        value=client.email,
                        placeholder="votre.email@exemple.com"
                    )

                    message_personnalise = st.text_area(
                        "💬 Message personnalisé (optionnel)",
                        placeholder="Message d'accompagnement pour le rapport...",
                        height=100
                    )

                with col2:
                    st.markdown("### 🚀 Actions")

                    if st.button("📄 Générer PDF", type="primary", use_container_width=True):
                        generer_et_telecharger_rapport(client, patrimoine, analyseur)

                    if st.button("📧 Envoyer par Email", type="secondary", use_container_width=True):
                        if email_destinataire:
                            envoyer_rapport_par_email(client, patrimoine, analyseur, email_destinataire,
                                                      message_personnalise)
                        else:
                            st.error("⚠️ Veuillez saisir une adresse email")

                    if st.button("👀 Aperçu", use_container_width=True):
                        afficher_apercu_rapport(client, patrimoine, analyseur)

            # Aperçu du contenu
            with st.expander("📖 Table des Matières", expanded=True):
                st.markdown(f"""
                **ANALYSE PATRIMONIALE PROFESSIONNELLE**  
                *{client.prenom} {client.nom} - {datetime.now().strftime('%B %Y')}*

                ---

                **1. SYNTHÈSE EXÉCUTIVE**
                - Profil patrimonial : {analyseur.profil_type.nom}
                - Patrimoine net : {ratios['patrimoine_net']:,.0f} €
                - Objectifs prioritaires : {len(client.objectifs)} identifiés

                **2. DIAGNOSTIC PATRIMONIAL DÉTAILLÉ**
                - Analyse de la répartition actuelle
                - Ratios clés et benchmarks sectoriels
                - Points forts et axes d'amélioration

                **3. RECOMMANDATIONS STRATÉGIQUES** 
                - {len(recommandations)} recommandations personnalisées
                - Plan d'action priorisé par impact
                - Calendrier de mise en œuvre sur 24 mois

                **4. PROJECTIONS ET SCÉNARIOS**
                - Simulations Monte Carlo sur {client.delai_objectif} ans
                - Analyse des risques (VaR 5%)
                - Allocation optimale selon profil

                **5. ANNEXES TECHNIQUES**
                - Méthodologie de calcul
                - Glossaire financier
                - Sources et références
                """)

            # Extrait du diagnostic
            with st.expander("📊 Extrait - Diagnostic Patrimonial"):
                points_forts = generer_points_forts(ratios, client)
                points_attention = generer_points_attention(ratios, client)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Points Forts Identifiés**")
                    for point in points_forts[:3]:
                        st.markdown(f"• {point}")

                with col2:
                    st.markdown("**⚠️ Points d'Attention**")
                    for point in points_attention[:3]:
                        st.markdown(f"• {point}")

            # Extrait des recommandations
            with st.expander("🎯 Extrait - Top 3 Recommandations"):
                for i, rec in enumerate(recommandations[:3], 1):
                    st.markdown(f"""
                    **{i}. {rec['titre']}** *[{rec['type']}]*

                    📝 **Analyse :** {rec['analyse'][:100]}...

                    🎯 **Action :** {rec['action']}

                    💰 **Impact estimé :** {rec['impact_financier']:,.0f} € / an

                    ---
                    """)

        # Fonctions utilitaires supplémentaires
        @st.cache_data
        def calculer_allocation_optimale_avancee(client: ClientProfile, profil_type: ProfileType) -> Dict:
            """Calcule l'allocation optimale selon le profil avancé"""

            # Base selon le profil type
            allocation_base = {
                'actions': profil_type.allocation_actions,
                'obligations': profil_type.allocation_obligations,
                'monetaire': profil_type.allocation_monetaire,
                'alternatif': profil_type.allocation_alternatif
            }

            # Ajustements selon l'âge et les objectifs
            if client.age < 30:
                allocation_base['actions'] += 10
                allocation_base['monetaire'] -= 10
            elif client.age > 60:
                allocation_base['actions'] -= 10
                allocation_base['obligations'] += 10

            # Ajustements selon les objectifs
            if "Investir en bourse" in client.objectifs:
                allocation_base['actions'] += 5
            if "Préparer la retraite" in client.objectifs and client.age > 45:
                allocation_base['obligations'] += 5
                allocation_base['actions'] -= 5

            # Normalisation à 100%
            total = sum(allocation_base.values())
            for key in allocation_base:
                allocation_base[key] = max(5, min(80, allocation_base[key] * 100 / total))

            return allocation_base

        def generer_insights_ia(client: ClientProfile, ratios: Dict, profil_type: ProfileType) -> List[str]:
            """Génère des insights intelligents basés sur l'IA d'analyse patrimoniale"""

            insights = []

            # Analyse de l'âge et du cycle de vie
            if client.age < 35 and ratios['exposition_risque'] < 0.4:
                insights.append(
                    f"🎯 À {client.age} ans, vous pourriez optimiser votre potentiel de croissance long terme en augmentant votre exposition aux marchés actions.")

            # Analyse familiale
            if client.enfants > 0 and ratios['ratio_liquidite'] < 4:
                insights.append(
                    f"👨‍👩‍👧‍👦 Avec {client.enfants} enfant(s), une épargne de précaution renforcée (4-6 mois) est recommandée pour faire face aux imprévus familiaux.")

            # Analyse professionnelle
            if client.secteur_activite == "Fonction publique" and ratios['utilisation_pea'] < 0.3:
                insights.append(
                    "🏛️ Votre stabilité professionnelle dans la fonction publique vous permet d'optimiser l'enveloppe PEA pour sa fiscalité avantageuse.")

            # Analyse de revenus
            revenus_totaux = client.revenu_annuel + client.revenus_passifs
            if revenus_totaux > 80000 and ratios['potentiel_per'] > 0:
                insights.append(
                    f"💰 Avec {revenus_totaux:,.0f}€ de revenus, l'optimisation fiscale via PER pourrait vous faire économiser plusieurs milliers d'euros d'impôts.")

            # Analyse de patrimoine
            if ratios['patrimoine_net'] > 500000 and client.enfants > 0:
                insights.append(
                    "🏛️ Votre patrimoine justifie une réflexion sur l'optimisation de la transmission familiale (donations, démembrement...).")

            # Analyse comportementale
            if client.tolerance_risque == "Très prudent" and client.age < 40:
                insights.append(
                    "⚖️ Votre jeune âge vous offre un avantage temps considérable. Une prise de risque mesurée pourrait significativement accroître votre patrimoine futur.")

            return insights[:4]  # Limite à 4 insights pour éviter la surcharge

        # Correction des bugs identifiés dans le code original
        def corriger_calculs_patrimoine(patrimoine: PatrimoineData) -> Dict:
            """Calculs patrimoniaux corrigés et harmonisés"""

            # Épargne liquide (disponible immédiatement)
            epargne_liquide = patrimoine.epargne_courante + patrimoine.livret_a + patrimoine.ldds + patrimoine.cel

            # Épargne réglementée (incluant PEL)
            epargne_reglementee = epargne_liquide + patrimoine.pel

            # Assurance-vie totale
            assurance_vie_total = patrimoine.assurance_vie_euro + patrimoine.assurance_vie_uc

            # Plans d'épargne retraite
            per_total = patrimoine.per_individuel + patrimoine.per_entreprise

            # PEA total
            pea_total = patrimoine.pea + patrimoine.pea_pme

            # Investissements financiers (hors immobilier)
            patrimoine_financier = (
                    epargne_reglementee + assurance_vie_total + pea_total +
                    patrimoine.cto + per_total + patrimoine.crypto +
                    patrimoine.or_physique + patrimoine.scpi + patrimoine.fonds_euros
            )

            # Patrimoine immobilier
            patrimoine_immobilier = patrimoine.immobilier_residence + patrimoine.immobilier_locatif

            # Dettes totales
            dettes_totales = (
                    patrimoine.credit_immobilier + patrimoine.credit_conso +
                    patrimoine.autres_dettes + patrimoine.pret_famille
            )

            # Patrimoine brut et net
            patrimoine_brut = patrimoine_financier + patrimoine_immobilier
            patrimoine_net = patrimoine_brut - dettes_totales

            return {
                'epargne_liquide': epargne_liquide,
                'epargne_reglementee': epargne_reglementee,
                'assurance_vie_total': assurance_vie_total,
                'per_total': per_total,
                'pea_total': pea_total,
                'patrimoine_financier': patrimoine_financier,
                'patrimoine_immobilier': patrimoine_immobilier,
                'dettes_totales': dettes_totales,
                'patrimoine_brut': patrimoine_brut,
                'patrimoine_net': patrimoine_net
            }

        def page_analyse_experte():
            if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
                st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
                return
            client = st.session_state.client
            patrimoine = st.session_state.patrimoine
            analyseur = AnalyseurPatrimoine(client, patrimoine)
            ratios = analyseur.calculer_ratios_avances()

            st.header("📊 Analyse Patrimoniale Experte")

            # Profil identifié
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h3>🎯 Profil Patrimonial Identifié</h3>
                    <h2>{analyseur.profil_type.nom}</h2>
                    <p>Caractéristiques principales :</p>
                    <ul>
                    {''.join([f'<li>{carac}</li>' for carac in analyseur.profil_type.caracteristiques])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #28a745;">{ratios['patrimoine_net'] / 1000:.0f}k€</div>
                    <div class="metric-label">Patrimoine Net</div>
                </div>
                """, unsafe_allow_html=True)

            # KPIs avancés
            st.subheader("📈 Indicateurs Clés de Performance")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                color = "#28a745" if ratios['ratio_liquidite'] >= 3 else "#dc3545" if ratios[
                                                                                          'ratio_liquidite'] < 2 else "#ffc107"
                st.metric("💧 Liquidité", f"{ratios['ratio_liquidite']:.1f} mois", delta="3-6 mois recommandés")

            with col2:
                color = "#28a745" if ratios['taux_epargne'] > 0.15 else "#ffc107" if ratios[
                                                                                         'taux_epargne'] > 0.10 else "#dc3545"
                st.metric("💰 Taux d'Épargne", f"{ratios['taux_epargne'] * 100:.0f}%", delta="15% minimum")

            with col3:
                st.metric("⚖️ Diversification", f"{ratios['diversification_supports'] * 100:.0f}%",
                          delta="80% optimal")

            with col4:
                exposition_cible = analyseur.profil_type.allocation_actions / 100
                delta_exposition = ratios['exposition_risque'] - exposition_cible
                st.metric("📈 Exposition Risque", f"{ratios['exposition_risque'] * 100:.0f}%",
                          delta=f"{delta_exposition * 100:+.0f}% vs cible")

            with col5:
                st.metric("🏛️ Ratio P/R", f"{ratios['ratio_patrimoine_revenu']:.1f}x",
                          delta="Multiple des revenus")

            # Graphiques d'analyse
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Graphique de répartition patrimoniale
                fig_repartition = creer_graphique_repartition_avance(patrimoine)
                st.plotly_chart(fig_repartition, use_container_width=True)

            with col2:
                # Graphique radar du profil
                fig_radar = creer_graphique_radar_avance(ratios, analyseur.profil_type)
                st.plotly_chart(fig_radar, use_container_width=True)

            # Analyse comparative par rapport aux benchmarks
            st.subheader("📊 Positionnement par Rapport aux Benchmarks")

            exposition_cible = analyseur.profil_type.allocation_actions / 100
            benchmarks_data = {
                'Indicateur': ['Épargne de précaution', 'Taux d\'épargne', 'Exposition actions', 'Diversification'],
                'Votre situation': [f"{ratios['ratio_liquidite']:.1f} mois", f"{ratios['taux_epargne'] * 100:.0f}%",
                                    f"{ratios['exposition_risque'] * 100:.0f}%",
                                    f"{ratios['diversification_supports'] * 100:.0f}%"],
                'Benchmark': ['3-6 mois', '15%', f"{analyseur.profil_type.allocation_actions:.0f}%", '80%'],
                'Évaluation': [
                    '✅ Excellent' if ratios['ratio_liquidite'] >= 3 else '⚠️ À améliorer' if ratios[
                                                                                                 'ratio_liquidite'] >= 2 else '❌ Insuffisant',
                    '✅ Excellent' if ratios['taux_epargne'] > 0.15 else '⚠️ Moyen' if ratios[
                                                                                          'taux_epargne'] > 0.10 else '❌ Faible',
                    '✅ Adapté' if abs(ratios['exposition_risque'] - exposition_cible) < 0.1 else '⚠️ À ajuster',
                    '✅ Bon' if ratios['diversification_supports'] > 0.6 else '⚠️ À améliorer'
                ]
            }

            df_benchmarks = pd.DataFrame(benchmarks_data)
            st.dataframe(df_benchmarks, hide_index=True, use_container_width=True)

        def creer_graphique_repartition_avance(patrimoine: PatrimoineData) -> go.Figure:
            """Crée un graphique de répartition patrimoniale avancé"""

            # Calcul des catégories
            liquidites = patrimoine.epargne_courante + patrimoine.livret_a + patrimoine.ldds + patrimoine.cel
            epargne_reglementee = patrimoine.pel
            assurance_vie = patrimoine.assurance_vie_euro + patrimoine.assurance_vie_uc
            pea_total = patrimoine.pea + patrimoine.pea_pme
            per_total = patrimoine.per_individuel + patrimoine.per_entreprise
            immobilier = patrimoine.immobilier_residence + patrimoine.immobilier_locatif
            alternatifs = patrimoine.crypto + patrimoine.or_physique + patrimoine.scpi

            categories = ['Liquidités', 'Épargne Réglementée', 'Assurance-Vie', 'PEA', 'CTO', 'PER', 'Immobilier',
                          'Alternatifs']
            valeurs = [liquidites, epargne_reglementee, assurance_vie, pea_total, patrimoine.cto, per_total, immobilier,
                       alternatifs]

            # Filtrer les valeurs nulles
            categories_filtered = [cat for cat, val in zip(categories, valeurs) if val > 0]
            valeurs_filtered = [val for val in valeurs if val > 0]

            if not valeurs_filtered:
                fig = go.Figure()
                fig.add_annotation(text="Aucune donnée patrimoniale", x=0.5, y=0.5, showarrow=False)
                return fig

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            fig = go.Figure(data=[go.Pie(
                labels=categories_filtered,
                values=valeurs_filtered,
                hole=0.4,
                marker_colors=colors[:len(categories_filtered)],
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Montant: %{value:,.0f} €<br>Part: %{percent}<extra></extra>'
            )])

            fig.update_layout(
                title="🏛️ Répartition Patrimoniale Détaillée",
                showlegend=True,
                height=400,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )

            return fig

        def creer_graphique_radar_avance(ratios: Dict, profil_type: ProfileType) -> go.Figure:
            """Crée un graphique radar comparant la situation actuelle au profil cible"""

            categories = ['Liquidité<br>(3-6 mois)', 'Épargne<br>(>15%)', 'Diversification<br>(>60%)',
                          'Exposition Risque<br>(profil)', 'Optimisation<br>Fiscale']

            # Normalisation des valeurs sur 100
            valeurs_actuelles = [
                min(100, ratios['ratio_liquidite'] * 20),  # 5 mois = 100%
                min(100, ratios['taux_epargne'] * 500),  # 20% = 100%
                ratios['diversification_supports'] * 100,
                100 - abs(ratios['exposition_risque'] - profil_type.allocation_actions / 100) * 200,  # Écart à la cible
                (ratios['utilisation_pea'] + ratios['utilisation_av']) * 50  # Utilisation enveloppes
            ]

            # Profil cible (optimal)
            valeurs_cibles = [80, 80, 80, 100, 80]

            fig = go.Figure()

            # Situation actuelle
            fig.add_trace(go.Scatterpolar(
                r=valeurs_actuelles,
                theta=categories,
                fill='toself',
                name='Situation Actuelle',
                line=dict(color='#2a5298', width=2),
                fillcolor='rgba(42, 82, 152, 0.25)'
            ))

            # Profil cible
            fig.add_trace(go.Scatterpolar(
                r=valeurs_cibles,
                theta=categories,
                fill='toself',
                name='Profil Optimal',
                line=dict(color='#28a745', width=2, dash='dash'),
                fillcolor='rgba(40, 167, 69, 0.15)'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%')
                ),
                showlegend=True,
                title="🎯 Analyse Comparative - Profil Patrimonial",
                height=400
            )

            return fig

        def page_recommandations_ia():
            if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
                st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
                return

            client = st.session_state.client
            patrimoine = st.session_state.patrimoine
            analyseur = AnalyseurPatrimoine(client, patrimoine)
            recommandations = analyseur.generer_recommandations_expertes()

            st.header("🎯 Recommandations d'Expert Personnalisées")

            # Score global et résumé
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                score_global = np.mean([r['priorite'] for r in recommandations]) * 10 if recommandations else 50
                couleur_score = "#28a745" if score_global > 70 else "#ffc107" if score_global > 50 else "#dc3545"

                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Score d'Optimisation Patrimoniale</h3>
                    <h1 style="color: {couleur_score}; margin: 0; font-size: 3em;">{score_global:.0f}/100</h1>
                    <p style="margin: 10px 0 0 0; color: #666;">Potentiel d'amélioration identifié par notre IA</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                nb_urgent = len([r for r in recommandations if r['type'] == 'URGENT'])
                st.metric("🚨 Actions Urgentes", nb_urgent, delta="À traiter immédiatement")

            with col3:
                impact_total = sum([r['impact_financier'] for r in recommandations])
                st.metric("💰 Impact Financier", f"{impact_total:,.0f} €", delta="Bénéfice estimé annuel")

            st.markdown("---")

            # Affichage des recommandations avec design professionnel
            for i, rec in enumerate(recommandations):
                couleurs_type = {
                    'URGENT': '#dc3545',
                    'STRATEGIQUE': '#fd7e14',
                    'REEQUILIBRAGE': '#6f42c1',
                    'FISCALITE': '#20c997',
                    'DIVERSIFICATION': '#17a2b8',
                    'TRANSMISSION': '#6c757d'
                }

                couleur = couleurs_type.get(rec['type'], '#6c757d')

                with st.expander(f"#{i + 1} • {rec['titre']}", expanded=(i < 2)):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Badges et catégorie
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span style="background: {couleur}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8em; font-weight: bold;">{rec['type']}</span>
                            <span style="margin-left: 1rem; color: #666; font-size: 0.9em;">{rec['categorie']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Analyse
                        st.markdown("**🔍 Analyse :**")
                        st.markdown(rec['analyse'])

                        # Action recommandée
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%); 
                                   border-left: 4px solid #2196f3; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
                            <strong>🎯 Action recommandée :</strong><br>
                            {rec['action']}
                        </div>
                        """, unsafe_allow_html=True)

                        # Bénéfice attendu
                        st.markdown("**✨ Bénéfice attendu :**")
                        st.markdown(f"<span style='color: #28a745; font-weight: 500;'>{rec['benefice']}</span>",
                                    unsafe_allow_html=True)

                        # Impact et délai
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666; margin-top: 1rem;">
                            <span><strong>💰 Impact :</strong> {rec['impact_financier']:,.0f} € / an</span>
                            <span><strong>⏱️ Délai :</strong> {rec['delai']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Indicateur de priorité
                        priority_pct = rec['priorite'] * 10
                        st.metric("Priorité", f"{rec['priorite']}/10", delta=f"{priority_pct:.0f}%")
                        st.progress(priority_pct / 100)

                        if st.button(f"📋 Détails", key=f"detail_{i}"):
                            st.info(
                                f"💡 Cette recommandation fait partie de votre stratégie {rec['categorie'].lower()}.")

                        # Barre de progression pour la priorité
                        st.progress(priority_pct / 100)

                        if st.button(f"📋 Détails", key=f"detail_{i}"):
                            st.info(
                                f"💡 Cette recommandation fait partie de votre stratégie {rec['categorie'].lower()}. "
                                f"Elle est classée en priorité {rec['priorite']} selon votre profil {analyseur.profil_type.nom}.")

        def page_projections_avancees():
            if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
                st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
                return

            client = st.session_state.client
            patrimoine = st.session_state.patrimoine
            analyseur = AnalyseurPatrimoine(client, patrimoine)
            ratios = analyseur.calculer_ratios_avances()

            st.header("📈 Projections et Simulations Avancées")

            # Paramètres de simulation
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⚙️ Paramètres de Simulation")

                versements_mensuels = st.number_input(
                    "Versements mensuels futurs (€)",
                    min_value=0,
                    value=int(ratios.get('patrimoine_financier', 50000) * 0.01),
                    step=50
                )

                inflation = st.slider("📈 Inflation annuelle (%)", 1.0, 4.0, 2.0, 0.1) / 100

                horizon_projection = st.slider("🎯 Horizon de projection (années)", 5, 30, client.delai_objectif)

            with col2:
                st.subheader("📊 Scénarios de Rendement")

                # Scénarios basés sur le profil type
                scenarios = {
                    "Conservateur": {"rendement": 0.025, "volatilite": 0.05},
                    "Équilibré": {"rendement": 0.055, "volatilite": 0.12},
                    "Dynamique": {"rendement": 0.075, "volatilite": 0.18}
                }

                for nom, params in scenarios.items():
                    st.metric(
                        f"{nom}",
                        f"{params['rendement'] * 100:.1f}%",
                        delta=f"Vol: {params['volatilite'] * 100:.0f}%"
                    )

            # Calculs de projections
            patrimoine_initial = ratios.get('patrimoine_financier', 0)

            if patrimoine_initial > 0:
                # Simulations Monte Carlo pour chaque scénario
                with st.spinner("🔄 Calcul des projections Monte Carlo..."):
                    projections = {}

                    for nom, params in scenarios.items():
                        sim = simuler_monte_carlo_avance(
                            patrimoine_initial,
                            versements_mensuels * 12,
                            params['rendement'],
                            params['volatilite'],
                            horizon_projection,
                            inflation
                        )
                        projections[nom] = sim

                # Graphique des projections
                fig_projections = creer_graphique_projections_avance(projections, horizon_projection)
                st.plotly_chart(fig_projections, use_container_width=True)

                # Tableau de synthèse
                st.subheader("📊 Synthèse des Projections")

                synthese_data = []
                for nom, proj in projections.items():
                    synthese_data.append({
                        'Scénario': nom,
                        'Médiane (finale)': f"{proj['percentiles']['p50'][-1]:,.0f} €",
                        'Cas favorable (75%)': f"{proj['percentiles']['p75'][-1]:,.0f} €",
                        'Cas défavorable (25%)': f"{proj['percentiles']['p25'][-1]:,.0f} €",
                        'Multiplication capital': f"x{proj['percentiles']['p50'][-1] / patrimoine_initial:.1f}",
                        'Objectif atteint': "✅" if proj['percentiles']['p50'][-1] > patrimoine_initial * 2 else "⚠️"
                    })

                df_synthese = pd.DataFrame(synthese_data)
                st.dataframe(df_synthese, hide_index=True, use_container_width=True)

                # Analyse des risques VaR (CORRIGÉE)
                st.subheader("⚠️ Analyse des Risques (Value at Risk)")

                col1, col2, col3 = st.columns(3)

                for i, (nom, proj) in enumerate(projections.items()):
                    with [col1, col2, col3][i]:
                        # CORRECTION DU CALCUL VaR
                        valeurs_finales = proj['percentiles']['p50']  # Toutes les valeurs
                        patrimoine_final = valeurs_finales[-1]  # Valeur finale
                        var_5 = np.percentile(valeurs_finales, 5)  # 5e percentile

                        # Calcul correct de la perte
                        if patrimoine_final > patrimoine_initial:
                            perte_max = max(0, (patrimoine_initial - var_5) / patrimoine_initial * 100)
                        else:
                            perte_max = abs((patrimoine_final - patrimoine_initial) / patrimoine_initial * 100)

                        couleur = "#28a745" if perte_max < 10 else "#ffc107" if perte_max < 20 else "#dc3545"

                        st.markdown(f"""
                                 <div class="metric-card" style="border-left-color: {couleur};">
                                     <h4>{nom}</h4>
                                     <div class="metric-value" style="color: {couleur};">-{perte_max:.1f}%</div>
                                     <div class="metric-label">Perte maximale (VaR 5%)</div>
                                     <p style="font-size: 0.8em; margin-top: 0.5rem; color: #666;">
                                         Dans 95% des cas, les pertes ne dépasseront pas ce niveau
                                     </p>
                                 </div>
                             """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Veuillez d'abord saisir votre patrimoine pour voir les projections.")
# Fermeture de la fonction page_projections_avancees()
        # (le code VaR que vous avez montré se termine ici)

# Ajoutez ces fonctions manquantes :

def page_patrimoine_detaille():
    st.header("💼 Patrimoine Détaillé")

    if 'client' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord compléter votre profil client.")
        return

    # Onglets pour organiser les actifs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Épargne & Liquidités", "📈 Investissements", "🏠 Immobilier", "💳 Dettes"])

    with tab1:
        st.subheader("💰 Épargne et Liquidités")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Comptes courants et épargne disponible**")
            epargne_courante = st.number_input("Compte courant (€)", min_value=0, value=0, step=100,
                                               key="epargne_courante")
            livret_a = st.number_input("Livret A (€)", min_value=0, value=0, step=100, key="livret_a")
            ldds = st.number_input("LDDS (€)", min_value=0, value=0, step=100, key="ldds")

        with col2:
            st.markdown("**Épargne réglementée**")
            cel = st.number_input("CEL (€)", min_value=0, value=0, step=100, key="cel")
            pel = st.number_input("PEL (€)", min_value=0, value=0, step=100, key="pel")
            fonds_euros = st.number_input("Fonds euros (€)", min_value=0, value=0, step=100, key="fonds_euros")

    with tab2:
        st.subheader("📈 Investissements Financiers")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Assurance-vie**")
            assurance_vie_euro = st.number_input("AV Fonds euros (€)", min_value=0, value=0, step=500,
                                                 key="assurance_vie_euro")
            assurance_vie_uc = st.number_input("AV Unités de compte (€)", min_value=0, value=0, step=500,
                                               key="assurance_vie_uc")

            st.markdown("**Plans d'épargne retraite**")
            per_individuel = st.number_input("PER individuel (€)", min_value=0, value=0, step=500, key="per_individuel")
            per_entreprise = st.number_input("PER entreprise (€)", min_value=0, value=0, step=500, key="per_entreprise")

        with col2:
            st.markdown("**Comptes titres et PEA**")
            pea = st.number_input("PEA (€)", min_value=0, value=0, step=500, key="pea")
            pea_pme = st.number_input("PEA-PME (€)", min_value=0, value=0, step=500, key="pea_pme")
            cto = st.number_input("Compte-titres ordinaire (€)", min_value=0, value=0, step=500, key="cto")

            st.markdown("**Investissements alternatifs**")
            scpi = st.number_input("SCPI (€)", min_value=0, value=0, step=1000, key="scpi")
            crypto = st.number_input("Cryptomonnaies (€)", min_value=0, value=0, step=100, key="crypto")
            or_physique = st.number_input("Or physique (€)", min_value=0, value=0, step=500, key="or_physique")

    with tab3:
        st.subheader("🏠 Patrimoine Immobilier")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Biens immobiliers**")
            immobilier_residence = st.number_input("Résidence principale (€)", min_value=0, value=0, step=5000,
                                                   key="immobilier_residence")
            immobilier_locatif = st.number_input("Immobilier locatif (€)", min_value=0, value=0, step=5000,
                                                 key="immobilier_locatif")

        with col2:
            st.markdown("**Estimation automatique**")
            if st.button("🏠 Estimer ma résidence"):
                st.info("💡 Utilisez les sites comme SeLoger, MeilleursAgents ou DVF pour estimer votre bien")

    with tab4:
        st.subheader("💳 Dettes et Crédits")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Crédits immobiliers**")
            credit_immobilier = st.number_input("Crédit immobilier restant (€)", min_value=0, value=0, step=1000,
                                                key="credit_immobilier")

            st.markdown("**Autres dettes**")
            credit_conso = st.number_input("Crédit consommation (€)", min_value=0, value=0, step=500,
                                           key="credit_conso")

        with col2:
            st.markdown("**Dettes diverses**")
            autres_dettes = st.number_input("Autres dettes (€)", min_value=0, value=0, step=500, key="autres_dettes")
            pret_famille = st.number_input("Prêt famille/amis (€)", min_value=0, value=0, step=500, key="pret_famille")

    # Résumé en temps réel
    st.markdown("---")
    st.subheader("📊 Résumé Patrimonial")

    # Calculs automatiques
    total_liquidites = epargne_courante + livret_a + ldds + cel
    total_epargne_reg = pel + fonds_euros
    total_investissements = (assurance_vie_euro + assurance_vie_uc + pea + pea_pme +
                             cto + per_individuel + per_entreprise + scpi + crypto + or_physique)
    total_immobilier = immobilier_residence + immobilier_locatif
    total_dettes = credit_immobilier + credit_conso + autres_dettes + pret_famille

    patrimoine_brut = total_liquidites + total_epargne_reg + total_investissements + total_immobilier
    patrimoine_net = patrimoine_brut - total_dettes

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💧 Liquidités", f"{total_liquidites:,.0f} €")
    with col2:
        st.metric("📈 Investissements", f"{total_investissements:,.0f} €")
    with col3:
        st.metric("🏛️ Patrimoine Brut", f"{patrimoine_brut:,.0f} €")
    with col4:
        st.metric("💎 Patrimoine Net", f"{patrimoine_net:,.0f} €", delta=f"-{total_dettes:,.0f} € dettes")

    # Bouton de sauvegarde
    if st.button("💾 Sauvegarder le Patrimoine", type="primary", key="save_patrimoine"):
        if sauvegarder_patrimoine():
            st.success("✅ Patrimoine sauvegardé avec succès!")
            st.balloons()


def page_analyse_experte():
    if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
        return

    client = st.session_state.client
    patrimoine = st.session_state.patrimoine
    analyseur = AnalyseurPatrimoine(client, patrimoine)
    ratios = analyseur.calculer_ratios_avances()

    st.header("📊 Analyse Patrimoniale Experte")
    st.session_state.progress = 75

    # Profil identifié
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div class="success-card">
            <h3>🎯 Profil Patrimonial Identifié</h3>
            <h2>{analyseur.profil_type.nom}</h2>
            <p>Caractéristiques principales :</p>
            <ul>
            {''.join([f'<li>{carac}</li>' for carac in analyseur.profil_type.caracteristiques])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #28a745;">{ratios['patrimoine_net'] / 1000:.0f}k€</div>
            <div class="metric-label">Patrimoine Net</div>
        </div>
        """, unsafe_allow_html=True)

    # KPIs avancés
    st.subheader("📈 Indicateurs Clés de Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💧 Liquidité", f"{ratios['ratio_liquidite']:.1f} mois", delta="3-6 mois recommandés")

    with col2:
        st.metric("💰 Taux d'Épargne", f"{ratios['taux_epargne'] * 100:.0f}%", delta="15% minimum")

    with col3:
        st.metric("⚖️ Diversification", f"{ratios['diversification_supports'] * 100:.0f}%", delta="80% optimal")

    with col4:
        exposition_cible = analyseur.profil_type.allocation_actions / 100
        delta_exposition = ratios['exposition_risque'] - exposition_cible
        st.metric("📈 Exposition Risque", f"{ratios['exposition_risque'] * 100:.0f}%",
                  delta=f"{delta_exposition * 100:+.0f}% vs cible")

    with col5:
        st.metric("🏛️ Ratio P/R", f"{ratios['ratio_patrimoine_revenu']:.1f}x", delta="Multiple des revenus")

    # Graphiques d'analyse
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique de répartition patrimoniale
        fig_repartition = creer_graphique_repartition_avance(patrimoine)
        st.plotly_chart(fig_repartition, use_container_width=True)

    with col2:
        # Graphique radar du profil
        fig_radar = creer_graphique_radar_avance(ratios, analyseur.profil_type)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Analyse comparative
    st.subheader("📊 Positionnement par Rapport aux Benchmarks")

    exposition_cible = analyseur.profil_type.allocation_actions / 100
    benchmarks_data = {
        'Indicateur': ['Épargne de précaution', 'Taux d\'épargne', 'Exposition actions', 'Diversification'],
        'Votre situation': [f"{ratios['ratio_liquidite']:.1f} mois", f"{ratios['taux_epargne'] * 100:.0f}%",
                            f"{ratios['exposition_risque'] * 100:.0f}%",
                            f"{ratios['diversification_supports'] * 100:.0f}%"],
        'Benchmark': ['3-6 mois', '15%', f"{analyseur.profil_type.allocation_actions:.0f}%", '80%'],
        'Évaluation': [
            '✅ Excellent' if ratios['ratio_liquidite'] >= 3 else '⚠️ À améliorer' if ratios[
                                                                                         'ratio_liquidite'] >= 2 else '❌ Insuffisant',
            '✅ Excellent' if ratios['taux_epargne'] > 0.15 else '⚠️ Moyen' if ratios[
                                                                                  'taux_epargne'] > 0.10 else '❌ Faible',
            '✅ Adapté' if abs(ratios['exposition_risque'] - exposition_cible) < 0.1 else '⚠️ À ajuster',
            '✅ Bon' if ratios['diversification_supports'] > 0.6 else '⚠️ À améliorer'
        ]
    }

    df_benchmarks = pd.DataFrame(benchmarks_data)
    st.dataframe(df_benchmarks, hide_index=True, use_container_width=True)


def page_recommandations_ia():
    if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
        return

    client = st.session_state.client
    patrimoine = st.session_state.patrimoine
    analyseur = AnalyseurPatrimoine(client, patrimoine)
    recommandations = analyseur.generer_recommandations_expertes()

    st.header("🎯 Recommandations d'Expert Personnalisées")

    # Score global et résumé
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        score_global = np.mean([r['priorite'] for r in recommandations]) * 10 if recommandations else 50
        couleur_score = "#28a745" if score_global > 70 else "#ffc107" if score_global > 50 else "#dc3545"

        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Score d'Optimisation Patrimoniale</h3>
            <h1 style="color: {couleur_score}; margin: 0; font-size: 3em;">{score_global:.0f}/100</h1>
            <p style="margin: 10px 0 0 0; color: #666;">Potentiel d'amélioration identifié par notre IA</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        nb_urgent = len([r for r in recommandations if r['type'] == 'URGENT'])
        st.metric("🚨 Actions Urgentes", nb_urgent, delta="À traiter immédiatement")

    with col3:
        impact_total = sum([r['impact_financier'] for r in recommandations])
        st.metric("💰 Impact Financier", f"{impact_total:,.0f} €", delta="Bénéfice estimé annuel")

    st.markdown("---", unsafe_allow_html=True)

    for i, rec in enumerate(recommandations):
        couleurs_type = {
            'URGENT': '#dc3545',
            'STRATEGIQUE': '#fd7e14',
            'REEQUILIBRAGE': '#6f42c1',
            'FISCALITE': '#20c997',
            'DIVERSIFICATION': '#17a2b8',
            'TRANSMISSION': '#6c757d'
        }

        couleur = couleurs_type.get(rec['type'], '#6c757d')

        with st.expander(f"#{i + 1} • {rec['titre']}", expanded=(i < 2)):
            col1, col2 = st.columns([3, 1])

            with col1:
                # Badges et catégorie
                st.markdown(f"""
                  <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                      <span style="background: {couleur}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8em; font-weight: bold;">{rec['type']}</span>
                      <span style="margin-left: 1rem; color: #666; font-size: 0.9em;">{rec['categorie']}</span>
                  </div>
                  """, unsafe_allow_html=True)

                # Analyse
                st.markdown("**🔍 Analyse :**")
                st.markdown(rec['analyse'])

                # Action recommandée
                st.markdown(f"""
                  <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%); 
                             border-left: 4px solid #2196f3; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
                      <strong>🎯 Action recommandée :</strong><br>
                      {rec['action']}
                  </div>
                  """, unsafe_allow_html=True)

                # Bénéfice attendu
                st.markdown("**✨ Bénéfice attendu :**")
                st.markdown(f"<span style='color: #28a745; font-weight: 500;'>{rec['benefice']}</span>",
                            unsafe_allow_html=True)

                # Impact et délai
                st.markdown(f"""
                  <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666; margin-top: 1rem;">
                      <span><strong>💰 Impact :</strong> {rec['impact_financier']:,.0f} € / an</span>
                      <span><strong>⏱️ Délai :</strong> {rec['delai']}</span>
                  </div>
                  """, unsafe_allow_html=True)

            with col2:
                # Indicateur de priorité
                priority_pct = rec['priorite'] * 10
                st.metric("Priorité", f"{rec['priorite']}/10", delta=f"{priority_pct:.0f}%")
                st.progress(priority_pct / 100)

                if st.button(f"📋 Détails", key=f"detail_{i}"):
                    st.info(f"💡 Cette recommandation fait partie de votre stratégie {rec['categorie'].lower()}.")

def page_projections_avancees():
   if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
       st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
       return

   client = st.session_state.client
   patrimoine = st.session_state.patrimoine
   analyseur = AnalyseurPatrimoine(client, patrimoine)
   ratios = analyseur.calculer_ratios_avances()

   st.header("📈 Projections et Simulations Avancées")

   # Paramètres de simulation
   col1, col2 = st.columns(2)

   with col1:
       st.subheader("⚙️ Paramètres de Simulation")

       versements_mensuels = st.number_input(
           "Versements mensuels futurs (€)",
           min_value=0,
           value=int(ratios.get('patrimoine_financier', 50000) * 0.01),
           step=50
       )

       inflation = st.slider("📈 Inflation annuelle (%)", 1.0, 4.0, 2.0, 0.1) / 100

       horizon_projection = st.slider("🎯 Horizon de projection (années)", 5, 30, client.delai_objectif)

   with col2:
       st.subheader("📊 Scénarios de Rendement")

       # Scénarios basés sur le profil type
       scenarios = {
           "Conservateur": {"rendement": 0.025, "volatilite": 0.05},
           "Équilibré": {"rendement": 0.055, "volatilite": 0.12},
           "Dynamique": {"rendement": 0.075, "volatilite": 0.18}
       }

       for nom, params in scenarios.items():
           st.metric(
               f"{nom}",
               f"{params['rendement']*100:.1f}%",
               delta=f"Vol: {params['volatilite']*100:.0f}%"
           )

   # Calculs de projections
   patrimoine_initial = ratios.get('patrimoine_financier', 0)

   if patrimoine_initial > 0:
       # Simulations Monte Carlo pour chaque scénario
       with st.spinner("🔄 Calcul des projections Monte Carlo..."):
           projections = {}

           for nom, params in scenarios.items():
               sim = simuler_monte_carlo_avance(
                   patrimoine_initial,
                   versements_mensuels * 12,
                   params['rendement'],
                   params['volatilite'],
                   horizon_projection,
                   inflation
               )
               projections[nom] = sim

       # Graphique des projections
       fig_projections = creer_graphique_projections_avance(projections, horizon_projection)
       st.plotly_chart(fig_projections, use_container_width=True)

       # Tableau de synthèse
       st.subheader("📊 Synthèse des Projections")

       synthese_data = []
       for nom, proj in projections.items():
           synthese_data.append({
               'Scénario': nom,
               'Médiane (finale)': f"{proj['percentiles']['p50'][-1]:,.0f} €",
               'Cas favorable (75%)': f"{proj['percentiles']['p75'][-1]:,.0f} €",
               'Cas défavorable (25%)': f"{proj['percentiles']['p25'][-1]:,.0f} €",
               'Multiplication capital': f"x{proj['percentiles']['p50'][-1] / patrimoine_initial:.1f}",
               'Objectif atteint': "✅" if proj['percentiles']['p50'][-1] > patrimoine_initial * 2 else "⚠️"
           })

       df_synthese = pd.DataFrame(synthese_data)
       st.dataframe(df_synthese, hide_index=True, use_container_width=True)

       # Analyse des risques VaR
       st.subheader("⚠️ Analyse des Risques (Value at Risk)")

       col1, col2, col3 = st.columns(3)

       for i, (nom, proj) in enumerate(projections.items()):
           with [col1, col2, col3][i]:
               var_5 = np.percentile(proj['percentiles']['p50'], 5)
               perte_max = max(0, (patrimoine_initial - var_5) / patrimoine_initial * 100)

               couleur = "#28a745" if perte_max < 10 else "#ffc107" if perte_max < 20 else "#dc3545"

               st.markdown(f"""
                   <div class="metric-card" style="border-left-color: {couleur};">
                       <h4>{nom}</h4>
                       <div class="metric-value" style="color: {couleur};">-{perte_max:.1f}%</div>
                       <div class="metric-label">Perte maximale (VaR 5%)</div>
                       <p style="font-size: 0.8em; margin-top: 0.5rem; color: #666;">
                           Dans 95% des cas, les pertes ne dépasseront pas ce niveau
                       </p>
                   </div>
               """, unsafe_allow_html=True)
   else:
       st.warning("⚠️ Veuillez d'abord saisir votre patrimoine pour voir les projections.")

def page_rapport_professionnel():
   if 'client' not in st.session_state or 'patrimoine' not in st.session_state:
       st.warning("⚠️ Veuillez d'abord compléter votre profil et patrimoine.")
       return

   client = st.session_state.client
   patrimoine = st.session_state.patrimoine
   analyseur = AnalyseurPatrimoine(client, patrimoine)

   st.header("📄 Génération de Rapport Professionnel")
   st.session_state.progress = 100

   # Options du rapport
   col1, col2 = st.columns(2)

   with col1:
       st.subheader("⚙️ Configuration du Rapport")

       inclure_graphiques = st.checkbox("📊 Inclure les graphiques", value=True)
       inclure_projections = st.checkbox("📈 Inclure les projections", value=True)
       inclure_benchmarks = st.checkbox("📊 Inclure les benchmarks", value=True)

       format_rapport = st.selectbox(
           "📄 Format de rapport",
           ["PDF complet", "Résumé exécutif", "Analyse détaillée"],
           key="format_rapport"
       )

   with col2:
       st.subheader("📋 Contenu Inclus")

       elements_inclus = [
           "✅ Synthèse patrimoniale",
           "✅ Profil investisseur identifié",
           "✅ Diagnostic détaillé",
           "✅ Recommandations prioritaires",
           "✅ Plan d'action structuré",
           "✅ Calendrier de mise en œuvre"
       ]

       if inclure_projections:
           elements_inclus.append("✅ Projections Monte Carlo")
       if inclure_benchmarks:
           elements_inclus.append("✅ Comparaison benchmarks")

       for element in elements_inclus:
           st.markdown(element)

   # Aperçu des métriques clés
   st.subheader("📊 Aperçu du Rapport")

   ratios = analyseur.calculer_ratios_avances()
   recommandations = analyseur.generer_recommandations_expertes()

   col1, col2, col3, col4 = st.columns(4)

   with col1:
       st.metric("🏛️ Patrimoine Net", f"{ratios['patrimoine_net']:,.0f} €")

   with col2:
       st.metric("🎯 Recommandations", len(recommandations))

   with col3:
       impact_total = sum([r['impact_financier'] for r in recommandations])
       st.metric("💰 Impact Annuel", f"{impact_total:,.0f} €")

   with col4:
       st.metric("👤 Profil", analyseur.profil_type.nom)

   # Génération et téléchargement
   st.markdown("---")

   col1, col2 = st.columns([2, 1])

   with col1:
       email_destinataire = st.text_input(
           "📧 Email de réception du rapport",
           value=client.email,
           placeholder="votre.email@exemple.com"
       )

       message_personnalise = st.text_area(
           "💬 Message personnalisé (optionnel)",
           placeholder="Message d'accompagnement pour le rapport...",
           height=100
       )

   with col2:
       st.markdown("### 🚀 Actions")

       if st.button("📄 Générer HTML", type="primary", use_container_width=True):
           generer_et_telecharger_rapport(client, patrimoine, analyseur)

       if st.button("📧 Envoyer par Email", type="secondary", use_container_width=True):
           if email_destinataire:
               envoyer_rapport_par_email(client, patrimoine, analyseur, email_destinataire, message_personnalise)
           else:
               st.error("⚠️ Veuillez saisir une adresse email")

       if st.button("👀 Aperçu", use_container_width=True):
           afficher_apercu_rapport(client, patrimoine, analyseur)

def generer_et_telecharger_rapport(client: ClientProfile, patrimoine: PatrimoineData, analyseur: AnalyseurPatrimoine):
   """Génère et propose le téléchargement du rapport HTML"""

   with st.spinner("📊 Génération du rapport en cours..."):
       try:
           # Génération du HTML
           html_content = generer_rapport_pdf(client, patrimoine, analyseur)

           # Nom du fichier
           nom_fichier = f"Analyse_Patrimoniale_{client.prenom}_{client.nom}_{datetime.now().strftime('%Y%m%d')}.html"

           st.success("✅ Rapport généré avec succès!")

           # Bouton de téléchargement
           st.download_button(
               label="📥 Télécharger le Rapport HTML",
               data=html_content,
               file_name=nom_fichier,
               mime="text/html",
               type="primary"
           )

           # Statistiques du rapport
           col1, col2, col3 = st.columns(3)

           with col1:
               st.metric("📄 Format", "HTML")
           with col2:
               st.metric("📊 Sections", "6-8")
           with col3:
               st.metric("💾 Taille", f"{len(html_content) / 1024:.0f} KB")

       except Exception as e:
           st.error(f"❌ Erreur lors de la génération du rapport : {str(e)}")

def afficher_apercu_rapport(client: ClientProfile, patrimoine: PatrimoineData, analyseur: AnalyseurPatrimoine):
   """Affiche un aperçu du rapport"""

   st.subheader("👀 Aperçu du Rapport")

   ratios = analyseur.calculer_ratios_avances()
   recommandations = analyseur.generer_recommandations_expertes()

   # Aperçu du contenu
   with st.expander("📖 Table des Matières", expanded=True):
       st.markdown(f"""
       **ANALYSE PATRIMONIALE PROFESSIONNELLE**  
       *{client.prenom} {client.nom} - {datetime.now().strftime('%B %Y')}*

       ---

       **1. SYNTHÈSE EXÉCUTIVE**
       - Profil patrimonial : {analyseur.profil_type.nom}
       - Patrimoine net : {ratios['patrimoine_net']:,.0f} €
       - Objectifs prioritaires : {len(client.objectifs)} identifiés

       **2. DIAGNOSTIC PATRIMONIAL DÉTAILLÉ**
       - Analyse de la répartition actuelle
       - Ratios clés et benchmarks sectoriels
       - Points forts et axes d'amélioration

       **3. RECOMMANDATIONS STRATÉGIQUES** 
       - {len(recommandations)} recommandations personnalisées
       - Plan d'action priorisé par impact
       - Calendrier de mise en œuvre sur 24 mois
       """)

   # Extrait des recommandations
   with st.expander("🎯 Extrait - Top 3 Recommandations"):
       for i, rec in enumerate(recommandations[:3], 1):
           st.markdown(f"""
           **{i}. {rec['titre']}** *[{rec['type']}]*

           📝 **Analyse :** {rec['analyse'][:100]}...

           🎯 **Action :** {rec['action']}

           💰 **Impact estimé :** {rec['impact_financier']:,.0f} € / an

           ---
           """)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Une erreur inattendue s'est produite : {str(e)}")
        st.info("💡 Veuillez actualiser la page ou contacter le support technique.")
