package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestGetClosestVowel(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("u", GetClosestVowel("yogurt"))
    assert.Equal("u", GetClosestVowel("full"))
    assert.Equal("", GetClosestVowel("easy"))
    assert.Equal("", GetClosestVowel("eAsy"))
    assert.Equal("", GetClosestVowel("ali"))
    assert.Equal("a", GetClosestVowel("bad"))
    assert.Equal("o", GetClosestVowel("most"))
    assert.Equal("", GetClosestVowel("ab"))
    assert.Equal("", GetClosestVowel("ba"))
    assert.Equal("", GetClosestVowel("quick"))
    assert.Equal("i", GetClosestVowel("anime"))
    assert.Equal("", GetClosestVowel("Asia"))
    assert.Equal("o", GetClosestVowel("Above"))
}
