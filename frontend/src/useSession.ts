import { useEffect, useState, useSyncExternalStore } from 'react';
import { Image } from 'react-native';

import * as api from './api';
import { MatchSession } from './session';

export function useSession() {
  const [session] = useState(() => new MatchSession(api));
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [preview, setPreview] = useState<{ file: File; uri: string } | null>(
    null,
  );

  useEffect(() => {
    const file = state.file;
    if (!file) return;
    const uri = URL.createObjectURL(file);
    let active = true;
    setPreview({ file, uri });
    Image.getSize(uri, (width, height) => {
      if (active) session.setImageSize(file, { width, height });
    });
    return () => {
      active = false;
      URL.revokeObjectURL(uri);
    };
  }, [session, state.file]);

  useEffect(() => () => session.cancel(), [session]);

  return {
    state,
    session,
    imageUri: preview?.file === state.file ? (preview?.uri ?? null) : null,
  };
}
