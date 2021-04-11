// Copyright (C) 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT

import { BoundariesActions, BoundariesActionTypes } from 'actions/boundaries-actions';
import { AuthActions, AuthActionTypes } from 'actions/auth-actions';
import { ShortcutsActions, ShortcutsActionsTypes } from 'actions/shortcuts-actions';
import { KeyMap, KeyMapItem } from 'utils/mousetrap-react';
import { ShortcutsState } from './interfaces';

function formatShortcuts(shortcuts: KeyMapItem): string {
    const list: string[] = shortcuts.sequences as string[];
    return `[${list
        .map((shortcut: string): string => {
            let keys = shortcut.split('+');
            keys = keys.map((key: string): string => `${key ? key[0].toUpperCase() : key}${key.slice(1)}`);
            keys = keys.join('+').split(/\s/g);
            keys = keys.map((key: string): string => `${key ? key[0].toUpperCase() : key}${key.slice(1)}`);
            return keys.join(' ');
        })
        .join(', ')}]`;
}

const defaultKeyMap = ({
    SWITCH_SHORTCUTS: {
        name: 'Show shortcuts',
        description: 'Open/hide the list of available shortcuts',
        sequences: ['f1'],
        action: 'keydown',
    },
    SWITCH_SETTINGS: {
        name: 'Show settings',
        description: 'Open/hide settings dialog',
        sequences: ['f2'],
        action: 'keydown',
    },

    SWITCH_ALL_LOCK: {
        name: 'Lock/unlock all objects',
        description: 'Change locked state for all objects in the side bar',
        sequences: ['t l'],
        action: 'keydown',
    },
    SWITCH_LOCK: {
        name: 'Lock/unlock an object',
        description: 'Change locked state for an active object',
        sequences: ['l'],
        action: 'keydown',
    },
    SWITCH_ALL_HIDDEN: {
        name: 'Hide/show all objects',
        description: 'Change hidden state for objects in the side bar',
        sequences: ['t h'],
        action: 'keydown',
    },
    SWITCH_HIDDEN: {
        name: 'Hide/show an object',
        description: 'Change hidden state for an active object',
        sequences: ['h'],
        action: 'keydown',
    },
    SWITCH_OCCLUDED: {
        name: 'Switch occluded',
        description: 'Change occluded property for an active object',
        sequences: ['q', '/'],
        action: 'keydown',
    },
    SWITCH_KEYFRAME: {
        name: 'Switch keyframe',
        description: 'Change keyframe property for an active track',
        sequences: ['k'],
        action: 'keydown',
    },
    SWITCH_OUTSIDE: {
        name: 'Switch outside',
        description: 'Change outside property for an active track',
        sequences: ['o'],
        action: 'keydown',
    },
    DELETE_OBJECT: {
        name: 'Delete object',
        description: 'Delete an active object. Use shift to force delete of locked objects',
        sequences: ['del', 'shift+del'],
        action: 'keydown',
    },
    TO_BACKGROUND: {
        name: 'To background',
        description: 'Put an active object "farther" from the user (decrease z axis value)',
        sequences: ['-', '_'],
        action: 'keydown',
    },
    TO_FOREGROUND: {
        name: 'To foreground',
        description: 'Put an active object "closer" to the user (increase z axis value)',
        sequences: ['+', '='],
        action: 'keydown',
    },
    COPY_SHAPE: {
        name: 'Copy shape',
        description: 'Copy shape to CVAT internal clipboard',
        sequences: ['ctrl+c'],
        action: 'keydown',
    },
    PROPAGATE_OBJECT: {
        name: 'Propagate object',
        description: 'Make a copy of the object on the following frames',
        sequences: ['ctrl+b'],
        action: 'keydown',
    },
    NEXT_KEY_FRAME: {
        name: 'Next keyframe',
        description: 'Go to the next keyframe of an active track',
        sequences: ['r'],
        action: 'keydown',
    },
    PREV_KEY_FRAME: {
        name: 'Previous keyframe',
        description: 'Go to the previous keyframe of an active track',
        sequences: ['e'],
        action: 'keydown',
    },

    NEXT_ATTRIBUTE: {
        name: 'Next attribute',
        description: 'Go to the next attribute',
        sequences: ['down'],
        action: 'keydown',
    },
    PREVIOUS_ATTRIBUTE: {
        name: 'Previous attribute',
        description: 'Go to the previous attribute',
        sequences: ['up'],
        action: 'keydown',
    },
    NEXT_OBJECT: {
        name: 'Next object',
        description: 'Go to the next object',
        sequences: ['tab'],
        action: 'keydown',
    },
    PREVIOUS_OBJECT: {
        name: 'Previous object',
        description: 'Go to the previous object',
        sequences: ['shift+tab'],
        action: 'keydown',
    },

    PASTE_SHAPE: {
        name: 'Paste shape',
        description: 'Paste a shape from internal CVAT clipboard',
        sequences: ['ctrl+v'],
        action: 'keydown',
    },
    SWITCH_DRAW_MODE: {
        name: 'Draw mode',
        description:
            'Repeat the latest procedure of drawing with the same parameters (shift to redraw an existing shape)',
        sequences: ['shift+n', 'n'],
        action: 'keydown',
    },
    OPEN_REVIEW_ISSUE: {
        name: 'Open an issue',
        description: 'Create a new issues in the review workspace',
        sequences: ['n'],
        action: 'keydown',
    },
    SWITCH_MERGE_MODE: {
        name: 'Merge mode',
        description: 'Activate or deactivate mode to merging shapes',
        sequences: ['m'],
        action: 'keydown',
    },
    SWITCH_SPLIT_MODE: {
        name: 'Split mode',
        description: 'Activate or deactivate mode to splitting shapes',
        sequences: ['alt+m'],
        action: 'keydown',
    },
    SWITCH_GROUP_MODE: {
        name: 'Group mode',
        description: 'Activate or deactivate mode to grouping shapes',
        sequences: ['g'],
        action: 'keydown',
    },
    RESET_GROUP: {
        name: 'Reset group',
        description: 'Reset group for selected shapes (in group mode)',
        sequences: ['shift+g'],
        action: 'keyup',
    },
    CANCEL: {
        name: 'Cancel',
        description: 'Cancel any active canvas mode',
        sequences: ['esc'],
        action: 'keydown',
    },
    CLOCKWISE_ROTATION: {
        name: 'Rotate clockwise',
        description: 'Change image angle (add 90 degrees)',
        sequences: ['ctrl+r'],
        action: 'keydown',
    },
    ANTICLOCKWISE_ROTATION: {
        name: 'Rotate anticlockwise',
        description: 'Change image angle (substract 90 degrees)',
        sequences: ['ctrl+shift+r'],
        action: 'keydown',
    },

    SAVE_JOB: {
        name: 'Save the job',
        description: 'Send all changes of annotations to the server',
        sequences: ['ctrl+s'],
        action: 'keydown',
    },
    UNDO: {
        name: 'Undo action',
        description: 'Cancel the latest action related with objects',
        sequences: ['ctrl+z'],
        action: 'keydown',
    },
    REDO: {
        name: 'Redo action',
        description: 'Cancel undo action',
        sequences: ['ctrl+shift+z', 'ctrl+y'],
        action: 'keydown',
    },
    NEXT_FRAME: {
        name: 'Next frame',
        description: 'Go to the next frame',
        sequences: ['f'],
        action: 'keydown',
    },
    PREV_FRAME: {
        name: 'Previous frame',
        description: 'Go to the previous frame',
        sequences: ['d'],
        action: 'keydown',
    },
    FORWARD_FRAME: {
        name: 'Forward frame',
        description: 'Go forward with a step',
        sequences: ['v'],
        action: 'keydown',
    },
    BACKWARD_FRAME: {
        name: 'Backward frame',
        description: 'Go backward with a step',
        sequences: ['c'],
        action: 'keydown',
    },
    SEARCH_FORWARD: {
        name: 'Search forward',
        description: 'Search the next frame that satisfies to the filters',
        sequences: ['right'],
        action: 'keydown',
    },
    SEARCH_BACKWARD: {
        name: 'Search backward',
        description: 'Search the previous frame that satisfies to the filters',
        sequences: ['left'],
        action: 'keydown',
    },
    PLAY_PAUSE: {
        name: 'Play/pause',
        description: 'Start/stop automatic changing frames',
        sequences: ['space'],
        action: 'keydown',
    },
    FOCUS_INPUT_FRAME: {
        name: 'Focus input frame',
        description: 'Focus on the element to change the current frame',
        sequences: ['~'],
        action: 'keydown',
    },
    SWITCH_AUTOMATIC_BORDERING: {
        name: 'Switch automatic bordering',
        description: 'Switch automatic bordering for polygons and polylines during drawing/editing',
        sequences: ['ctrl'],
        action: 'keydown',
    },
    CHANGE_OBJECT_COLOR: {
        name: 'Change color',
        description: 'Set the next color for an activated shape',
        sequences: ['enter'],
        action: 'keydown',
    },
    TOGGLE_LAYOUT_GRID: {
        name: 'Toggle layout grid',
        description: 'The grid is used to UI development',
        sequences: ['ctrl+alt+enter'],
        action: 'keydown',
    },
    SWITCH_LABEL: {
        name: 'Switch label',
        description: 'Changes a label for an activated object or for the next drawn object if no objects are activated',
        sequences: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'].map((val: string): string => `ctrl+${val}`),
        action: 'keydown',
    },
} as any) as KeyMap;

const defaultState: ShortcutsState = {
    visibleShortcutsHelp: false,
    keyMap: defaultKeyMap,
    normalizedKeyMap: Object.keys(defaultKeyMap).reduce((acc: Record<string, string>, key: string) => {
        const normalized = formatShortcuts(defaultKeyMap[key]);
        acc[key] = normalized;
        return acc;
    }, {}),
};

export default (state = defaultState, action: ShortcutsActions | BoundariesActions | AuthActions): ShortcutsState => {
    switch (action.type) {
        case ShortcutsActionsTypes.SWITCH_SHORTCUT_DIALOG: {
            return {
                ...state,
                visibleShortcutsHelp: !state.visibleShortcutsHelp,
            };
        }
        case BoundariesActionTypes.RESET_AFTER_ERROR:
        case AuthActionTypes.LOGOUT_SUCCESS: {
            return { ...defaultState };
        }
        default: {
            return state;
        }
    }
};
