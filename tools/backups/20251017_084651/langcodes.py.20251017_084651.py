"""
Language code utilities and mappings for multilingual ASR and translation
Provides normalization between different language code standards and model-specific formats

This module implements the "langauto" detection and normalization system.
"""

import re
from typing import Dict, Optional, Tuple, List
import warnings


# ISO 639-1 to ISO 639-3 mappings for common languages;
ISO_639_1_TO_3: Dict[str, str] = {
    'ar': 'ara', 'de': 'deu', 'en': 'eng', 'es': 'spa', 'fr': 'fra',
    'hi': 'hin', 'it': 'ita', 'ja': 'jpn', 'ko': 'kor', 'nl': 'nld',
    'pl': 'pol', 'pt': 'por', 'ru': 'rus', 'sv': 'swe', 'tr': 'tur',
    'uk': 'ukr', 'vi': 'vie', 'zh': 'zho', 'fa': 'fas', 'he': 'heb',
    'id': 'ind', 'th': 'tha', 'cs': 'ces', 'da': 'dan', 'el': 'ell',
    'fi': 'fin', 'hu': 'hun', 'no': 'nor', 'ro': 'ron', 'sk': 'slk',
    'sl': 'slv', 'bg': 'bul', 'hr': 'hrv', 'et': 'est', 'lv': 'lav',
    'lt': 'lit', 'mt': 'mlt', 'ca': 'cat', 'eu': 'eus', 'gl': 'glg'
};

# Language names to ISO 639-1;
LANGUAGE_NAMES_TO_ISO: Dict[str, str] = {
    'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
    'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'chinese': 'zh',
    'japanese': 'ja', 'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi',
    'dutch': 'nl', 'swedish': 'sv', 'polish': 'pl', 'turkish': 'tr',
    'ukrainian': 'uk', 'vietnamese': 'vi', 'persian': 'fa', 'hebrew': 'he',
    'indonesian': 'id', 'thai': 'th', 'czech': 'cs', 'danish': 'da',
    'greek': 'el', 'finnish': 'fi', 'hungarian': 'hu', 'norwegian': 'no',
    'romanian': 'ro', 'slovak': 'sk', 'slovenian': 'sl', 'bulgarian': 'bg',
    'croatian': 'hr', 'estonian': 'et', 'latvian': 'lv', 'lithuanian': 'lt',
    'maltese': 'mt', 'catalan': 'ca', 'basque': 'eu', 'galician': 'gl'
};

# NLLB language tags (script-aware);
NLLB_LANGUAGE_CODES: Dict[str, str] = {
    'ar': 'ara_Arab', 'de': 'deu_Latn', 'en': 'eng_Latn', 'es': 'spa_Latn',
    'fr': 'fra_Latn', 'hi': 'hin_Deva', 'it': 'ita_Latn', 'ja': 'jpn_Jpan',
    'ko': 'kor_Hang', 'nl': 'nld_Latn', 'pl': 'pol_Latn', 'pt': 'por_Latn',
    'ru': 'rus_Cyrl', 'sv': 'swe_Latn', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl',
    'vi': 'vie_Latn', 'zh': 'zho_Hans', 'fa': 'fas_Arab', 'he': 'heb_Hebr',
    'id': 'ind_Latn', 'th': 'tha_Thai', 'cs': 'ces_Latn', 'da': 'dan_Latn',
    'el': 'ell_Grek', 'fi': 'fin_Latn', 'hu': 'hun_Latn', 'no': 'nor_Latn',
    'ro': 'ron_Latn', 'sk': 'slk_Latn', 'sl': 'slv_Latn', 'bg': 'bul_Cyrl',
    'hr': 'hrv_Latn', 'et': 'est_Latn', 'lv': 'lav_Latn', 'lt': 'lit_Latn',
    'mt': 'mlt_Latn', 'ca': 'cat_Latn', 'eu': 'eus_Latn', 'gl': 'glg_Latn'
};

# Language display names;
LANGUAGE_DISPLAY_NAMES: Dict[str, str] = {
    'ar': 'Arabic', 'de': 'German', 'en': 'English', 'es': 'Spanish',
    'fr': 'French', 'hi': 'Hindi', 'it': 'Italian', 'ja': 'Japanese',
    'ko': 'Korean', 'nl': 'Dutch', 'pl': 'Polish', 'pt': 'Portuguese',
    'ru': 'Russian', 'sv': 'Swedish', 'tr': 'Turkish', 'uk': 'Ukrainian',
    'vi': 'Vietnamese', 'zh': 'Chinese', 'fa': 'Persian', 'he': 'Hebrew',
    'id': 'Indonesian', 'th': 'Thai', 'cs': 'Czech', 'da': 'Danish',
    'el': 'Greek', 'fi': 'Finnish', 'hu': 'Hungarian', 'no': 'Norwegian',
    'ro': 'Romanian', 'sk': 'Slovak', 'sl': 'Slovenian', 'bg': 'Bulgarian',
    'hr': 'Croatian', 'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian',
    'mt': 'Maltese', 'ca': 'Catalan', 'eu': 'Basque', 'gl': 'Galician'
};


def normalize_language_code( lang_input: str ) -> Optional[str]:
    """
    Normalize various language inputs to ISO 639-1 codes
    
    Args:
        lang_input: Language input (code, name, or variant)
        
    Returns:
        ISO 639-1 code if valid, None if unrecognized
    """
    if not lang_input:
        return None;
    
    # Convert to lowercase and strip;
    normalized = lang_input.lower().strip();
    
    # Handle special cases;
    if normalized in [ 'auto', 'detect', 'automatic' ]:
        return 'auto';
    
    # Check if already ISO 639-1;
    if len( normalized ) == 2 and normalized in ISO_639_1_TO_3:
        return normalized;
    
    # Check language names;
    if normalized in LANGUAGE_NAMES_TO_ISO:
        return LANGUAGE_NAMES_TO_ISO[normalized];
    
    # Check 3-letter codes;
    if len( normalized ) == 3:
        for iso1, iso3 in ISO_639_1_TO_3.items():
            if iso3 == normalized:
                return iso1;
    
    # Check NLLB format (e.g., 'spa_Latn' -> 'es');
    if '_' in normalized:
        base_code = normalized.split( '_' )[0];
        for iso1, iso3 in ISO_639_1_TO_3.items():
            if iso3 == base_code:
                return iso1;
    
    # Handle common alternatives;
    alternatives = {
        'zh-cn': 'zh', 'zh-hans': 'zh', 'mandarin': 'zh',
        'pt-br': 'pt', 'pt-pt': 'pt', 'brazilian': 'pt',
        'en-us': 'en', 'en-uk': 'en', 'american': 'en', 'british': 'en',
        'es-es': 'es', 'es-mx': 'es', 'castilian': 'es', 'mexican': 'es',
        'fr-fr': 'fr', 'fr-ca': 'fr', 'canadian': 'fr'
    };
    
    if normalized in alternatives:
        return alternatives[normalized];
    
    return None;


def get_nllb_code( lang_code: str ) -> Optional[str]:
    """
    Get NLLB language tag for a language code
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        NLLB language tag (e.g., 'eng_Latn') or None
    """
    normalized = normalize_language_code( lang_code );
    if normalized and normalized != 'auto':
        return NLLB_LANGUAGE_CODES.get( normalized );
    return None;


def get_display_name( lang_code: str ) -> str:
    """
    Get human-readable display name for a language code
    
    Args:
        lang_code: Language code
        
    Returns:
        Display name (e.g., 'Spanish') or the original code
    """
    normalized = normalize_language_code( lang_code );
    if normalized and normalized != 'auto':
        return LANGUAGE_DISPLAY_NAMES.get( normalized, lang_code );
    elif normalized == 'auto':
        return 'Auto-detect';
    return lang_code;


def validate_language_code( lang_code: str ) -> Tuple[bool, Optional[str]]:
    """
    Validate and normalize a language code
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        Tuple of (is_valid, normalized_code or error_message)
    """
    if not lang_code:
        return False, "Empty language code";
    
    normalized = normalize_language_code( lang_code );
    if normalized is None:
        # Suggest similar codes;
        suggestions = get_language_suggestions( lang_code );
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else "";
        return False, f"Unrecognized language code '{lang_code}'{suggestion_text}";
    
    return True, normalized;


def get_language_suggestions( lang_input: str ) -> List[str]:
    """
    Get language code suggestions for invalid input
    
    Args:
        lang_input: Invalid language input
        
    Returns:
        List of suggested valid codes
    """
    suggestions = [];
    normalized_input = lang_input.lower().strip();
    
    # Check partial matches in language names;
    for name, code in LANGUAGE_NAMES_TO_ISO.items():
        if normalized_input in name or name.startswith( normalized_input ):
            suggestions.append( f"{code} ({name.title()})" );
    
    # Check partial matches in display names;
    for code, display in LANGUAGE_DISPLAY_NAMES.items():
        if normalized_input in display.lower():
            suggestions.append( f"{code} ({display})" );
    
    return suggestions[:5];  # Limit to top 5 suggestions;


def get_supported_languages() -> Dict[str, str]:
    """
    Get all supported language codes with display names
    
    Returns:
        Dictionary mapping ISO codes to display names
    """
    return LANGUAGE_DISPLAY_NAMES.copy();


def is_rtl_language( lang_code: str ) -> bool:
    """
    Check if language is right-to-left
    
    Args:
        lang_code: Language code
        
    Returns:
        True if RTL language
    """
    rtl_languages = { 'ar', 'he', 'fa' };
    normalized = normalize_language_code( lang_code );
    return normalized in rtl_languages if normalized else False;


def get_whisper_language_name( lang_code: str ) -> Optional[str]:
    """
    Get Whisper-compatible language name for language code
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        Whisper language name or None for auto-detect
    """
    normalized = normalize_language_code( lang_code );
    if not normalized or normalized == 'auto':
        return None;  # Let Whisper auto-detect;
    
    # Map to Whisper-expected names;
    whisper_names = {
        'ar': 'arabic', 'de': 'german', 'en': 'english', 'es': 'spanish',
        'fr': 'french', 'hi': 'hindi', 'it': 'italian', 'ja': 'japanese',
        'ko': 'korean', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese',
        'ru': 'russian', 'sv': 'swedish', 'tr': 'turkish', 'uk': 'ukrainian',
        'vi': 'vietnamese', 'zh': 'chinese', 'fa': 'persian', 'he': 'hebrew',
        'id': 'indonesian', 'th': 'thai', 'cs': 'czech', 'da': 'danish',
        'el': 'greek', 'fi': 'finnish', 'hu': 'hungarian', 'no': 'norwegian',
        'ro': 'romanian', 'sk': 'slovak', 'sl': 'slovenian', 'bg': 'bulgarian',
        'hr': 'croatian', 'et': 'estonian', 'lv': 'latvian', 'lt': 'lithuanian',
        'ca': 'catalan', 'eu': 'basque', 'gl': 'galician'
    };
    
    return whisper_names.get( normalized );


class LanguageCodeError( ValueError ):
    """Exception raised for invalid language codes"""
    pass;


def require_valid_language( lang_code: str ) -> str:
    """
    Validate language code or raise exception
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        Normalized language code
        
    Raises:
        LanguageCodeError: If language code is invalid
    """
    is_valid, result = validate_language_code( lang_code );
    if not is_valid:
        raise LanguageCodeError( result );
    return result;