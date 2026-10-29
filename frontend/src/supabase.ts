import { createClient } from "@supabase/supabase-js";

// The anon key is public by design: Row Level Security denies it all data access, and the
// browser only uses it to sign in. All data goes through the backend with the user's token.
export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL as string,
  import.meta.env.VITE_SUPABASE_ANON_KEY as string,
  { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } },
);
