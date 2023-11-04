package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestIsPalindrome(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, IsPalindrome(""))
    assert.Equal(true, IsPalindrome("aba"))
    assert.Equal(true, IsPalindrome("aaaaa"))
    assert.Equal(false, IsPalindrome("zbcd"))
    assert.Equal(true, IsPalindrome("xywyx"))
    assert.Equal(false, IsPalindrome("xywyz"))
    assert.Equal(false, IsPalindrome("xywzx"))
}
