export const CREDIT = "Made by Pascal";

export function PrivacyNotice() {
  return (
    <>
      <p>
        <strong>What Paradigm saves:</strong> your username, a scrambled copy of your password, your assistant setup,
        your chats, the text of documents you upload, facts it remembers about you, links you share, and usage counts
        for rate limits. Temporary chats aren't saved.
      </p>
      <p>
        <strong>Who else handles your data:</strong> Groq receives your messages and document excerpts to write replies.
        A web search service receives search terms when the assistant searches the web. Neon stores your data, and
        Vercel runs the app. Voice typing and read aloud use your browser&apos;s speech features, and some browsers send
        audio to their own speech service.
      </p>
      <p>
        <strong>Your choices:</strong> in Settings, under Memory, you can see and delete what Paradigm remembers. Under
        Account, you can download your data or delete your account. Deleting removes your account, chats, setup,
        documents, memories and share links. Anyone with a share link can read that chat until you turn the link off.
      </p>
      <p>Don&apos;t enter passwords, payment details or other highly sensitive information in chats.</p>
    </>
  );
}
